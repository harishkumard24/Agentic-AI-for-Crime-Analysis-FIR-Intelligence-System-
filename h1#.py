import argparse
import os
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants — numeric-to-label mappings (explicit, no guesswork)
# ---------------------------------------------------------------------------

DOW_MAP = {
    0: "Sunday",
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
}

TOD_MAP = {
    0: "Morning",
    1: "Afternoon",
    2: "Evening",
    3: "Night",
    4: "Late Night",
}

MONTH_NAME_MAP = {
    1: "January", 2: "February", 3: "March",    4: "April",
    5: "May",     6: "June",     7: "July",      8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

WEEKEND_DAYS = {"Saturday", "Sunday"}


# ---------------------------------------------------------------------------
# Step 1 — Load raw CSV
# ---------------------------------------------------------------------------

def load_raw_csv(path: str) -> pd.DataFrame:
    print(f"\n[1/6] Reading CSV: {path}")
    try:
        df = pd.read_csv(path, encoding="utf-8", dtype=str)
    except UnicodeDecodeError:
        print("      UTF-8 failed, retrying with utf-8-sig ...")
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)

    print(f"      Loaded {len(df):,} rows x {len(df.columns)} columns")
    print(f"      Columns: {df.columns.tolist()}")
    return df


# ---------------------------------------------------------------------------
# Step 2 — General preprocessing
# ---------------------------------------------------------------------------

def _to_snake_case(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[\s\-/]+", "_", name)
    name = re.sub(r"[^\w]", "", name)
    name = re.sub(r"_+", "_", name)
    return name


def _clean_text(series: pd.Series) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def _safe_float_to_nullable_int(series: pd.Series) -> pd.array:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.astype("Int64")


def general_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[2/6] General preprocessing ...")

    # Snake-case column names
    df.columns = [_to_snake_case(c) for c in df.columns]

    # Replace literal "nan" / "NaN" strings introduced by dtype=str read
    df.replace({"nan": pd.NA, "NaN": pd.NA, "": pd.NA}, inplace=True)

    # Identify column categories
    text_cols    = ["crime_type", "location", "crime_description"]
    numeric_cols = ["day_of_week", "time_of_day", "day", "month", "year"]

    # Clean text columns
    for col in text_cols:
        if col in df.columns:
            df[col] = _clean_text(df[col])
            df.loc[df[col].str.strip() == "", col] = pd.NA

    # Convert numeric columns to nullable Int64
    for col in numeric_cols:
        if col in df.columns:
            df[col] = _safe_float_to_nullable_int(df[col])

    # Remove exact duplicate rows
    original_count = len(df)
    df.drop_duplicates(inplace=True)
    dupes_removed = original_count - len(df)
    df.reset_index(drop=True, inplace=True)

    # Stable record_id starting at 1
    df.insert(0, "record_id", range(1, len(df) + 1))

    print(f"      Input rows          : {original_count:,}")
    print(f"      Duplicates removed  : {dupes_removed:,}")
    print(f"      Rows after cleaning : {len(df):,}")
    print(f"      Columns             : {df.columns.tolist()}")

    null_counts = df.isnull().sum()
    print("\n      Null counts per column:")
    for col, cnt in null_counts.items():
        if cnt > 0:
            print(f"        {col:<25}: {cnt:,}")

    return df, dupes_removed


# ---------------------------------------------------------------------------
# Step 3 — Build SQL-ready dataset
# ---------------------------------------------------------------------------

def _derive_dow_label(series: pd.Series) -> pd.Series:
    return series.map(lambda v: DOW_MAP.get(int(v), None) if pd.notna(v) else None)


def _derive_tod_label(series: pd.Series) -> pd.Series:
    return series.map(lambda v: TOD_MAP.get(int(v), None) if pd.notna(v) else None)


def _derive_is_weekend(dow_label_series: pd.Series) -> pd.Series:
    def check(v):
        if pd.isna(v):
            return pd.NA
        return v in WEEKEND_DAYS
    return dow_label_series.map(check).astype("boolean")


def build_sql_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[3/6] Building SQL-ready dataset ...")
    sql = df.copy()

    # --- Validated day / month / year ----------------------------------
    # day: valid 1–31, month: valid 1–12, year: keep as-is (Int64)
    sql["day"]   = sql["day"].where(sql["day"].between(1, 31))
    sql["month"] = sql["month"].where(sql["month"].between(1, 12))

    # --- Numeric code preserved + human-readable mapped columns --------
    sql["day_of_week_code"] = sql["day_of_week"]
    sql["time_of_day_code"] = sql["time_of_day"]

    sql["day_of_week_label"] = _derive_dow_label(sql["day_of_week"])
    sql["time_of_day_label"] = _derive_tod_label(sql["time_of_day"])

    # --- Normalised filter columns (ASCII-safe lowercase) ---------------
    sql["crime_type_normalized"] = (
        sql["crime_type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace("nan", pd.NA)
    )
    sql["location_normalized"] = (
        sql["location"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace("nan", pd.NA)
    )

    # --- Calendar derivations ------------------------------------------
    sql["month_name"] = sql["month"].map(
        lambda v: MONTH_NAME_MAP.get(int(v), None) if pd.notna(v) else None
    )

    sql["quarter"] = sql["month"].map(
        lambda v: ((int(v) - 1) // 3 + 1) if pd.notna(v) else None
    ).astype("Int64")

    # --- Weekend flag ---------------------------------------------------
    sql["is_weekend"] = _derive_is_weekend(sql["day_of_week_label"])

    # --- Complete date flag + date_key ----------------------------------
    has_day   = sql["day"].notna()
    has_month = sql["month"].notna()
    has_year  = sql["year"].notna()
    sql["has_complete_date"] = (has_day & has_month & has_year).astype("boolean")

    def build_date_key(row):
        if row["has_complete_date"]:
            try:
                return f"{int(row['year']):04d}-{int(row['month']):02d}-{int(row['day']):02d}"
            except Exception:
                return None
        return None

    sql["date_key"] = sql.apply(build_date_key, axis=1)

    # --- Column order ---------------------------------------------------
    ordered_cols = [
        "record_id",
        "crime_type", "crime_type_normalized",
        "location",   "location_normalized",
        "day_of_week_code", "day_of_week_label",
        "time_of_day_code", "time_of_day_label",
        "day", "month", "month_name", "quarter", "year",
        "is_weekend", "has_complete_date", "date_key",
        "crime_description",
    ]
    sql = sql[[c for c in ordered_cols if c in sql.columns]]

    print(f"      SQL-ready rows    : {len(sql):,}")
    print(f"      SQL-ready columns : {sql.columns.tolist()}")
    return sql


# ---------------------------------------------------------------------------
# Step 4 — Build vector-ready dataset
# ---------------------------------------------------------------------------

def _safe_val(value, label: str) -> str:
    if pd.isna(value):
        return ""
    val = str(value).strip()
    if val.lower() == "nan" or val == "":
        return ""
    return f"{label}: {val}"


def build_fir_text(row: pd.Series) -> str:
    parts = [
        _safe_val(row.get("crime_type"),         "Crime Type"),
        _safe_val(row.get("location"),            "Location"),
        _safe_val(row.get("day_of_week_label"),   "Day of Week"),
        _safe_val(row.get("time_of_day_label"),   "Time of Day"),
        _safe_val(row.get("day"),                 "Day"),
        _safe_val(row.get("month_name"),          "Month"),
        _safe_val(row.get("year"),                "Year"),
        _safe_val(row.get("crime_description"),   "Description"),
    ]
    return "\n".join(p for p in parts if p)


def build_vector_dataset(sql_df: pd.DataFrame) -> pd.DataFrame:
    print("\n[4/6] Building vector-ready dataset ...")
    vec = sql_df[[
        "record_id",
        "crime_type", "location",
        "day_of_week_code", "day_of_week_label",
        "time_of_day_code", "time_of_day_label",
        "day", "month", "year",
        "crime_description",
        "month_name",
    ]].copy()

    vec["fir_text"] = sql_df.apply(build_fir_text, axis=1)

    empty_fir = (vec["fir_text"].str.strip() == "").sum()
    print(f"      Vector-ready rows    : {len(vec):,}")
    print(f"      Empty fir_text rows  : {empty_fir}")

    final_cols = [
        "record_id", "fir_text",
        "crime_type", "location",
        "day_of_week_code", "day_of_week_label",
        "time_of_day_code", "time_of_day_label",
        "day", "month", "year",
    ]
    return vec[[c for c in final_cols if c in vec.columns]]


# ---------------------------------------------------------------------------
# Step 5 — Save outputs
# ---------------------------------------------------------------------------

def save_outputs(sql_df: pd.DataFrame, vec_df: pd.DataFrame, output_dir: str):
    print(f"\n[5/6] Saving outputs to: {output_dir}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sql_csv_path = out / "sql_ready_fir.csv"
    vec_csv_path = out / "vector_ready_fir.csv"
    db_path      = out / "fir_relational.db"

    # Save CSVs with UTF-8-SIG to preserve Kannada in all editors
    sql_df.to_csv(sql_csv_path, index=False, encoding="utf-8-sig")
    print(f"      Saved: {sql_csv_path}")

    vec_df.to_csv(vec_csv_path, index=False, encoding="utf-8-sig")
    print(f"      Saved: {vec_csv_path}")

    # SQLite — pandas to_sql handles Unicode TEXT natively
    # Convert Int64 / boolean to Python-native types for sqlite3 compatibility
    sql_for_db = sql_df.copy()
    for col in sql_for_db.columns:
        if hasattr(sql_for_db[col], "dtype"):
            if str(sql_for_db[col].dtype) in ("Int64", "boolean"):
                sql_for_db[col] = sql_for_db[col].astype(object).where(
                    sql_for_db[col].notna(), other=None
                )

    conn = sqlite3.connect(str(db_path))
    sql_for_db.to_sql("fir_cases", conn, if_exists="replace", index=False)
    row_count = conn.execute("SELECT COUNT(*) FROM fir_cases").fetchone()[0]
    conn.close()
    print(f"      Saved: {db_path}  (table: fir_cases, {row_count:,} rows)")

    return sql_csv_path, vec_csv_path, db_path


# ---------------------------------------------------------------------------
# Step 6 — Quality report + profiling summary
# ---------------------------------------------------------------------------

def print_quality_report(sql_df: pd.DataFrame, vec_df: pd.DataFrame, dupes_removed: int):
    print("\n[6/6] Quality report")
    print("=" * 70)

    print("\n-- SQL-ready dataset preview (first 3 rows, selected cols) --")
    preview_cols = [
        "record_id", "crime_type", "location",
        "day_of_week_label", "time_of_day_label",
        "day", "month_name", "year", "date_key", "is_weekend",
    ]
    print(sql_df[[c for c in preview_cols if c in sql_df.columns]].head(3).to_string(index=False))

    print("\n-- Vector-ready dataset preview (first 2 rows, fir_text) --")
    for _, row in vec_df.head(2).iterrows():
        print(f"\n  record_id={row['record_id']}")
        print("  " + "\n  ".join(row["fir_text"].split("\n")))

    print("\n-- Null counts (SQL-ready) --")
    null_summary = sql_df.isnull().sum()
    for col, cnt in null_summary[null_summary > 0].items():
        print(f"  {col:<30}: {cnt:,}")

    print(f"\n-- Duplicates removed  : {dupes_removed:,}")
    print(f"   Final SQL rows      : {len(sql_df):,}")
    print(f"   Final vector rows   : {len(vec_df):,}")

    print("\n-- Profiling summary (unique value counts) --")
    profile = {
        "crime_type":         sql_df["crime_type"].nunique(),
        "location":           sql_df["location"].nunique(),
        "year":               sql_df["year"].nunique(),
        "month":              sql_df["month"].nunique(),
        "day_of_week_label":  sql_df["day_of_week_label"].nunique()
                              if "day_of_week_label" in sql_df.columns else "N/A",
        "time_of_day_label":  sql_df["time_of_day_label"].nunique()
                              if "time_of_day_label" in sql_df.columns else "N/A",
    }
    for k, v in profile.items():
        print(f"  {k:<25}: {v}")

    print("\n-- Top 10 crime types --")
    print(
        sql_df["crime_type"]
        .value_counts()
        .head(10)
        .to_string()
    )

    print("\n-- Year distribution --")
    year_dist = (
        sql_df["year"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print(year_dist)

    print("\n-- Day-of-week label distribution --")
    if "day_of_week_label" in sql_df.columns:
        print(
            sql_df["day_of_week_label"]
            .value_counts()
            .sort_index()
            .to_string()
        )

    print("\n-- Time-of-day label distribution --")
    if "time_of_day_label" in sql_df.columns:
        print(
            sql_df["time_of_day_label"]
            .value_counts()
            .sort_index()
            .to_string()
        )

    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Udupi FIR dataset for hybrid SQL + vector RAG pipeline."
    )
    parser.add_argument(
        "--input_csv",
        default="/mnt/data/UdupiCrimeData.csv",
        help="Path to raw input CSV (default: /mnt/data/UdupiCrimeData.csv)",
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory for output files (default: current directory)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.isfile(args.input_csv):
        print(f"ERROR: Input file not found: {args.input_csv}", file=sys.stderr)
        sys.exit(1)

    try:
        raw_df = load_raw_csv(args.input_csv)
        clean_df, dupes_removed = general_preprocessing(raw_df)
        sql_df  = build_sql_dataset(clean_df)
        vec_df  = build_vector_dataset(sql_df)
        save_outputs(sql_df, vec_df, args.output_dir)
        print_quality_report(sql_df, vec_df, dupes_removed)
        print("\nDone.")

    except Exception as exc:
        print(f"\nFATAL ERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

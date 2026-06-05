"""
APP13.py  —  FIR Intelligence System v3
Udupi District FIR Analysis Assistant

Architecture:
  ResponseStyleDetector → QueryParser → IntentClassifier
  → FIRAgent → [SQLiteManager | ChromaVectorStore | SarvamLLM]
  → StreamlitChatUI

Combines:
  - APP12 rich 7-intent routing + chat UI + response style system
  - First app backend transparency + expander debug visibility
  - Correct DB schema (fir_cases with actual columns only)
  - Sarvam API mode (no local HF model)
  - Session-based chat + query history
"""

import json
import os
import re
import sqlite3
import time
import traceback
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import requests
import streamlit as st
from chromadb.utils import embedding_functions

# ---------------------------------------------------------------------------
# Page config — MUST be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FIR Intelligence System",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"]   { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header     { visibility: hidden; }

.stApp                        { background-color: #f9f9fb; color: #111; }
.block-container              { padding-top: 2rem; padding-bottom: 5rem; max-width: 820px; }

/* Header */
.fir-header                   { padding: 1.2rem 0 0.25rem; }
.fir-header h1                {
    font-family: 'DM Mono', monospace;
    font-size: 1.2rem; font-weight: 500; color: #111;
    letter-spacing: 0.04em; border-bottom: 2px solid #111;
    padding-bottom: 0.6rem; margin: 0;
}
.fir-header p                 {
    font-family: 'DM Mono', monospace; font-size: 0.72rem;
    color: #999; letter-spacing: 0.02em; margin: 0.3rem 0 0;
}

/* Chat bubbles */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #ffffff; border: 1px solid #e4e6eb;
    border-radius: 16px 16px 4px 16px; padding: 0.8rem 1rem;
    margin: 0.4rem 0 0.4rem 2.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: #eef2ff; border: 1px solid #dce3fb;
    border-radius: 16px 16px 16px 4px; padding: 0.8rem 1rem;
    margin: 0.4rem 2.5rem 0.4rem 0;
    box-shadow: 0 1px 4px rgba(99,102,241,0.07);
}
[data-testid="chatAvatarIcon-user"]      { background: #e4e6eb !important; color: #444 !important; }
[data-testid="chatAvatarIcon-assistant"] { background: #6366f1 !important; color: #fff !important; }
[data-testid="stChatMessage"] p          { font-size: 0.92rem; line-height: 1.7; color: #1a1a1a; margin: 0; }

/* Trace bar */
.trace-bar {
    font-family: 'DM Mono', monospace; font-size: 0.68rem;
    color: #aaa; letter-spacing: 0.01em; line-height: 1.8;
    margin-top: 0.4rem;
}
.trace-bar span { color: #555; font-weight: 500; }

/* Expander */
[data-testid="stExpander"]           { border: 1px solid #e5e7eb !important; border-radius: 6px !important; background: #fafafa !important; margin-top: 0.4rem !important; }
[data-testid="stExpander"] summary   { font-family: 'DM Mono', monospace !important; font-size: 0.73rem !important; color: #aaa !important; letter-spacing: 0.02em !important; }
[data-testid="stExpander"] summary:hover { color: #555 !important; }

/* Detail blocks inside expander */
.sec-label {
    font-family: 'DM Mono', monospace; font-size: 0.64rem;
    color: #bbb; letter-spacing: 0.1em; text-transform: uppercase;
    margin: 0.65rem 0 0.2rem;
}
.detail-block {
    background: #f5f5f5; border: 1px solid #e8e8e8; border-radius: 4px;
    padding: 0.6rem 0.85rem; font-family: 'DM Mono', monospace;
    font-size: 0.72rem; color: #666; line-height: 1.6;
    word-break: break-all; white-space: pre-wrap;
}
.detail-block.err { color: #b44; }

/* Input */
[data-testid="stChatInput"] {
    border-radius: 12px !important; border: 1.5px solid #dce0ea !important;
    background: #ffffff !important; font-size: 0.92rem !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}
.stSpinner > div { border-top-color: #6366f1 !important; }

/* Empty state */
.empty-hint {
    text-align: center; padding: 2.5rem 1rem 1.5rem; color: #bbb;
}
.empty-hint p   { font-size: 0.86rem; line-height: 2; margin: 0; }
.empty-hint code {
    color: #888; background: #edf0f5;
    padding: 1px 6px; border-radius: 4px; font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SQLITE_DB_PATH   = "outputs/fir_relational.db"
CHROMA_PATH      = "outputs/chroma_store"
CHROMA_COLLECTION = "fir_documents"
EMBEDDING_MODEL   = "intfloat/multilingual-e5-base"
SARVAM_API_URL    = "https://api.sarvam.ai/v1/chat/completions"
SARVAM_API_KEY    = os.environ.get("SARVAM_API_KEY", "")
SARVAM_MODEL      = "sarvam-m"

MAX_LIST_ROWS     = 10
MAX_EVIDENCE_ROWS = 8
MAX_SNIPPET_LEN   = 240
TOP_K_CHROMA      = 5
TOP_K_ANALYTICS   = 8
DEFAULT_TOP_K     = 5

# ---------------------------------------------------------------------------
# Intent & Style Constants
# ---------------------------------------------------------------------------

INTENT_COUNT      = "COUNT"
INTENT_DESCRIBE   = "DESCRIBE"
INTENT_HYBRID     = "HYBRID"
INTENT_LIST       = "LIST"
INTENT_LOOKUP     = "LOOKUP"
INTENT_ANALYTICAL = "ANALYTICAL"
INTENT_SIMILAR    = "SIMILAR"
INTENT_OFF_TOPIC  = "OFF_TOPIC"

STYLE_LIST    = "LIST_STYLE"
STYLE_DETAIL  = "DETAIL_STYLE"
STYLE_SUMMARY = "SUMMARY_STYLE"
STYLE_REPORT  = "REPORT_STYLE"
STYLE_PATTERN = "PATTERN_STYLE"

# Answer mode labels (for trace)
AMODE_SQL_COUNT     = "SQL_COUNT"
AMODE_SQL_DIRECT    = "SQL_DIRECT"
AMODE_SQL_SUMMARY   = "SQL_SUMMARY"
AMODE_SQL_HYBRID    = "SQL_HYBRID"
AMODE_SQL_ANALYTICS = "SQL_ANALYTICS"
AMODE_SEM_RELATED   = "SEMANTIC_RELATED"
AMODE_SEM_SUMMARY   = "SEMANTIC_SUMMARY"
AMODE_STATIC        = "STATIC"
AMODE_LLM_SUMMARY   = "LLM_SUMMARY"

# ---------------------------------------------------------------------------
# Knowledge tables (schema-compatible)
# ---------------------------------------------------------------------------

MONTH_MAP: Dict[str, int] = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
    "jan":1,"feb":2,"mar":3,"apr":4,"jun":6,"jul":7,"aug":8,
    "sep":9,"oct":10,"nov":11,"dec":12,
    "ಜನವರಿ":1,"ಫೆಬ್ರವರಿ":2,"ಮಾರ್ಚ್":3,"ಏಪ್ರಿಲ್":4,"ಮೇ":5,"ಜೂನ್":6,
    "ಜುಲೈ":7,"ಆಗಸ್ಟ್":8,"ಸೆಪ್ಟೆಂಬರ್":9,"ಅಕ್ಟೋಬರ್":10,"ನವೆಂಬರ್":11,"ಡಿಸೆಂಬರ್":12,
}
MONTH_NUM_KN: Dict[int, str] = {
    1:"ಜನವರಿ",2:"ಫೆಬ್ರವರಿ",3:"ಮಾರ್ಚ್",4:"ಏಪ್ರಿಲ್",5:"ಮೇ",6:"ಜೂನ್",
    7:"ಜುಲೈ",8:"ಆಗಸ್ಟ್",9:"ಸೆಪ್ಟೆಂಬರ್",10:"ಅಕ್ಟೋಬರ್",11:"ನವೆಂಬರ್",12:"ಡಿಸೆಂಬರ್",
}

# Maps query keywords → crime_type_normalized values stored in DB
CRIME_KW: Dict[str, str] = {
    "ಹಲ್ಲೆ":"ಹಲ್ಲೆ",    "assault":"ಹಲ್ಲೆ",    "attack":"ಹಲ್ಲೆ",
    "ಕಳವು":"ಕಳವು",      "theft":"ಕಳವು",       "steal":"ಕಳವು",   "stolen":"ಕಳವು",
    "ಅಪಘಾತ":"ಅಪಘಾತ",   "accident":"ಅಪಘಾತ",  "crash":"ಅಪಘಾತ",
    "ಜುಗಾರಿ":"ಜುಗಾರಿ",  "gambling":"ಜುಗಾರಿ", "matka":"ಜುಗಾರಿ",  "ಮಟ್ಕಾ":"ಜುಗಾರಿ",
    "ಗಾಂಜಾ":"ಗಾಂಜಾ",    "ganja":"ಗಾಂಜಾ",     "drug":"ಗಾಂಜಾ",   "drugs":"ಗಾಂಜಾ",
    "ಕಾಣೆ":"ಕಾಣೆ",      "missing":"ಕಾಣೆ",
    "ಅಸ್ವಾಭಾವಿಕ":"ಅಸ್ವಾಭಾವಿಕ","unnatural":"ಅಸ್ವಾಭಾವಿಕ","death":"ಅಸ್ವಾಭಾವಿಕ",
    "ವಂಚನೆ":"ವಂಚನೆ",    "fraud":"ವಂಚನೆ",     "cheating":"ವಂಚನೆ",
    "robbery":"ದರೋಡೆ",  "ದರೋಡೆ":"ದರೋಡೆ",
}

# day_of_week_label values in DB are English (Saturday, Sunday, …)
DAY_MAP: Dict[str, str] = {
    "monday":"Monday",   "ಸೋಮವಾರ":"Monday",
    "tuesday":"Tuesday", "ಮಂಗಳವಾರ":"Tuesday",
    "wednesday":"Wednesday","ಬುಧವಾರ":"Wednesday",
    "thursday":"Thursday","ಗುರುವಾರ":"Thursday",
    "friday":"Friday",   "ಶುಕ್ರವಾರ":"Friday",
    "saturday":"Saturday","ಶನಿವಾರ":"Saturday",
    "sunday":"Sunday",   "ಭಾನುವಾರ":"Sunday",
}

# time_of_day_label values in DB: Morning, Afternoon, Evening, Night
TIME_MAP: Dict[str, str] = {
    "morning":"Morning",    "ಬೆಳಿಗ್ಗೆ":"Morning",   "ಮುಂಜಾನೆ":"Morning",
    "afternoon":"Afternoon","ಮಧ್ಯಾಹ್ನ":"Afternoon",
    "evening":"Evening",    "ಸಂಜೆ":"Evening",
    "night":"Night",        "ರಾತ್ರಿ":"Night",        "ನಿಶೆ":"Night",
}

LOCATION_NORM: Dict[str, str] = {
    "udupi":"ಉಡುಪಿ",       "manipal":"ಮಣಿಪಾಲ",
    "kundapura":"ಕುಂಧಾಪುರ","kundapur":"ಕುಂಧಾಪುರ",
    "brahmavar":"ಬ್ರಹ್ಮಾವರ","padubidri":"ಪಡುಬಿದ್ರಿ",
    "kapu":"ಕಾಪು",          "malpe":"ಮಲ್ಪೆ",
    "karkala":"ಕಾರ್ಕಳ",     "shirva":"ಶಿರ್ವಾ",
    "byndoor":"ಬೈಂದೂರು",    "gangoli":"ಗಂಗೊಳ್ಳಿ",
    "kollur":"ಕೊಲ್ಲೂರು",    "ambalapadi":"ಅಂಬಲಪಾಡಿ",
    "hebri":"ಹೆಬ್ರಿ",        "perdoor":"ಪೆರ್ಡೂರು",
    "kota":"ಕೋಟ",            "uchila":"ಉಚ್ಚಿಲ",
    "shankaranarayana":"ಶಂಕರನಾರಾಯಣ",
}
KNOWN_KN_LOCATIONS = list(LOCATION_NORM.values()) + [
    "ಉಡುಪಿ","ಮಣಿಪಾಲ","ಕುಂಧಾಪುರ","ಬ್ರಹ್ಮಾವರ","ಪಡುಬಿದ್ರಿ",
    "ಕಾಪು","ಮಲ್ಪೆ","ಕಾರ್ಕಳ","ಶಿರ್ವಾ","ಬೈಂದೂರು","ಗಂಗೊಳ್ಳಿ",
    "ಅಂಬಲಪಾಡಿ","ಹೆಬ್ರಿ","ಪೆರ್ಡೂರು","ಕೋಟ","ಉಚ್ಚಿಲ",
]

OUT_OF_DISTRICT = [
    "mangaluru","mangalore","ಮಂಗಳೂರು","bangalore","bengaluru","ಬೆಂಗಳೂರು",
    "mysore","mysuru","ಮೈಸೂರು","hubli","dharwad","ಹುಬ್ಬಳ್ಳಿ","belagavi",
    "shimoga","shivamogga","ಶಿವಮೊಗ್ಗ","hassan","ಹಾಸನ",
]

CRIME_INDICATORS = [
    "murder","kill","rape","robbery","kidnap","abduct","arson","harass",
    "stalk","extort","smuggl","ಕೊಲೆ","ಅತ್ಯಾಚಾರ","ಅಪಹರಣ","ಸುಲಿಗೆ",
]

FIR_KEYWORDS = [
    "fir","case","crime","assault","theft","murder","robbery","police",
    "ಪೊಲೀಸ್","ಕೇಸ್","ಅಪರಾಧ","ಕಳ್ಳತನ","ಹಲ್ಲೆ","ಕೊಲೆ","fraud","udupi",
    "ಉಡುಪಿ","ವಂಚನೆ","ದರೋಡೆ","ಅಪಘಾತ","accident","gambling","drug",
]

# ===========================================================================
# ResponseStyleDetector
# ===========================================================================

class ResponseStyleDetector:
    _SUPPRESS   = ["ಪಟ್ಟಿ ಬೇಡ","list ಬೇಡ","don't list","do not list","no list","without list"]
    _LIST_KW    = ["ಪಟ್ಟಿ","list","show all","give me all","fetch all","display all"]
    _SUMMARY_KW = ["summary","summarize","summarise","ಸಾರಾಂಶ","ಸಂಕ್ಷಿಪ್ತ","briefly","brief",
                   "concise","key points","quick summary","quickly explain","ಮುಖ್ಯ ಅಂಶ"]
    _PATTERN_KW = ["pattern","ಮಾದರಿ","ಸ್ವರೂಪ","general nature","ಸಾಮಾನ್ಯ ಸ್ವರೂಪ","ಸಾಮಾನ್ಯ ಮಾದರಿ"]
    _REPORT_KW  = ["report","ವರದಿ","findings","short report","ಚಿಕ್ಕ ವರದಿ"]
    _N_SENT_RE  = re.compile(r'\d+\s*(?:sentences?|ವಾಕ್ಯ)', re.IGNORECASE)

    @classmethod
    def detect(cls, query: str) -> str:
        ql = query.lower()
        suppress    = any(ph in ql for ph in cls._SUPPRESS)
        has_pattern = any(kw in ql for kw in cls._PATTERN_KW)
        has_summary = any(kw in ql for kw in cls._SUMMARY_KW)
        has_report  = any(kw in ql for kw in cls._REPORT_KW)
        has_n_sent  = bool(cls._N_SENT_RE.search(ql))
        wants_list  = (not suppress) and any(kw in ql for kw in cls._LIST_KW)

        if suppress or has_pattern or has_summary or has_n_sent or has_report:
            if has_pattern: return STYLE_PATTERN
            if has_report:  return STYLE_REPORT
            return STYLE_SUMMARY
        if wants_list:
            return STYLE_LIST
        return STYLE_DETAIL

    @classmethod
    def is_summary_style(cls, style: str) -> bool:
        return style in (STYLE_SUMMARY, STYLE_REPORT, STYLE_PATTERN)


# ===========================================================================
# QueryParser
# ===========================================================================

class QueryParser:

    @staticmethod
    def extract_year(query: str) -> Optional[int]:
        m = re.search(r'\b(20\d{2})\b', query)
        return int(m.group(1)) if m else None

    @staticmethod
    def extract_month(query: str) -> Tuple[Optional[int], Optional[str]]:
        ql = query.lower()
        for name, num in MONTH_MAP.items():
            is_kn = any('\u0C80' <= c <= '\u0CFF' for c in name)
            if is_kn:
                tokens = re.split(r'[\s,.\-/()\[\]]+', query)
                if name in tokens:
                    return num, MONTH_NUM_KN.get(num, name)
            else:
                if re.search(r'\b' + re.escape(name) + r'\b', ql):
                    return num, MONTH_NUM_KN.get(num, name)
        return None, None

    @staticmethod
    def extract_crime(query: str) -> Optional[str]:
        ql = query.lower()
        for kw, canonical in CRIME_KW.items():
            if kw in ql:
                return canonical
        return None

    @classmethod
    def extract_location(cls, query: str) -> Optional[str]:
        ql = query.lower()
        for eng, kn in LOCATION_NORM.items():
            if eng in ql:
                return kn
        for loc in KNOWN_KN_LOCATIONS:
            if loc in query:
                return loc
        return None

    @staticmethod
    def extract_day(query: str) -> Tuple[Optional[str], bool]:
        ql = query.lower()
        if "ವಾರಾಂತ್ಯ" in query or "weekend" in ql:
            return None, True
        for kw, label in DAY_MAP.items():
            if kw.lower() in ql:
                is_wknd = label in ("Saturday", "Sunday")
                return label, is_wknd
        return None, False

    @staticmethod
    def extract_time_of_day(query: str) -> Optional[str]:
        ql = query.lower()
        for kw, label in TIME_MAP.items():
            if kw.lower() in ql:
                return label
        return None

    @staticmethod
    def extract_record_id(query: str) -> Optional[str]:
        m = re.search(r'\b(REC\d+|record[_\s]?id[:\s]+(\S+))', query, re.IGNORECASE)
        if m:
            return m.group(2) or m.group(1)
        return None

    @classmethod
    def location_out_of_district(cls, query: str, loc: Optional[str]) -> bool:
        if loc: return False
        ql = query.lower()
        return any(city in ql for city in OUT_OF_DISTRICT)

    @classmethod
    def crime_unknown(cls, query: str, crime: Optional[str]) -> bool:
        if crime: return False
        ql = query.lower()
        return any(ind in ql for ind in CRIME_INDICATORS)

    @classmethod
    def parse(cls, query: str) -> Dict[str, Any]:
        year              = cls.extract_year(query)
        month, month_name = cls.extract_month(query)
        crime             = cls.extract_crime(query)
        location          = cls.extract_location(query)
        day_label, is_wknd= cls.extract_day(query)
        time_label        = cls.extract_time_of_day(query)
        record_id         = cls.extract_record_id(query)
        has_filters       = any([year, month, crime, location, day_label, is_wknd, time_label])
        return dict(
            year=year, month=month, month_name=month_name,
            crime=crime, location=location,
            day_label=day_label, is_weekend=is_wknd,
            time_label=time_label, record_id=record_id,
            has_filters=has_filters,
            unknown_location=cls.location_out_of_district(query, location),
            unknown_crime=cls.crime_unknown(query, crime),
        )


# ===========================================================================
# IntentClassifier
# ===========================================================================

_COUNT_KW      = ["ಎಷ್ಟು","ಸಂಖ್ಯೆ","ಒಟ್ಟು","ಎಣಿಕೆ","count","how many","total","number of"]
_DESCRIBE_KW   = ["ವಿವರ","ವಿವರಣೆ","ವಿವರಿಸಿ","ಮಾಹಿತಿ","ಹೇಳಿ","ತಿಳಿಸಿ",
                  "ಸಾರಾಂಶ","ಸಂಕ್ಷಿಪ್ತ","ಮಾದರಿ","ಸ್ವರೂಪ",
                  "describe","explain","details","information","tell me","about",
                  "summary","summarize","summarise","pattern","nature","report","overview","brief"]
_LIST_KW       = ["ಪಟ್ಟಿ","list","show all","give me all","fetch all","display all"]
_ANALYTICAL_KW = ["ಹೆಚ್ಚು","ಯಾವ ಸ್ಥಳ","ಯಾವ ವರ್ಷ","ಯಾವ ತಿಂಗಳ","ಯಾವ ದಿನ",
                  "hotspot","most","top","highest","which year","which location",
                  "which place","which crime","compare","ವಿಶ್ಲೇಷಣೆ","trend","ranked"]
_SIMILAR_KW    = ["ಇದೇ ರೀತಿ","ಸಮಾನ","similar","like this","resembles"]
_LOOKUP_KW     = ["ಪ್ರಕರಣ ನಂಬ್ರ","case number","record id","rec id"]
_HYBRID_CONN   = ["ಮತ್ತು ವಿವರ","ಮತ್ತು ಮಾಹಿತಿ","ಮತ್ತು ಸಾರಾಂಶ",
                  "and details","and summarize","and describe","and summary",
                  "also summarize","also describe"]

class IntentClassifier:

    @classmethod
    def classify(cls, query: str) -> Tuple[str, float, str]:
        ql    = query.lower()
        style = ResponseStyleDetector.detect(query)

        suppress_list = any(kw in ql for kw in ["ಪಟ್ಟಿ ಬೇಡ","don't list","no list"])
        is_summary    = ResponseStyleDetector.is_summary_style(style)
        is_list_style = (style == STYLE_LIST) and not suppress_list
        is_pattern    = any(kw in ql for kw in ["pattern","ಮಾದರಿ","ಸ್ವರೂಪ"])

        has_count      = any(kw in ql for kw in _COUNT_KW)
        has_desc       = any(kw in ql for kw in _DESCRIBE_KW) or is_summary
        has_list       = is_list_style and any(kw in ql for kw in _LIST_KW)
        has_analytical = any(kw in ql for kw in _ANALYTICAL_KW)
        has_similar    = any(kw in ql for kw in _SIMILAR_KW)
        has_lookup     = any(kw in ql for kw in _LOOKUP_KW)
        has_hybrid     = any(kw in ql for kw in _HYBRID_CONN)

        if is_pattern or is_summary:
            has_analytical = False

        if has_analytical:             return INTENT_ANALYTICAL, 0.95, "rule"
        if has_similar:                return INTENT_SIMILAR,    0.95, "rule"
        if has_lookup and not (is_summary or has_desc):
                                       return INTENT_LOOKUP,     0.90, "rule"
        if has_hybrid or (has_count and has_desc):
                                       return INTENT_HYBRID,     0.92, "rule"
        if has_count:                  return INTENT_COUNT,      0.95, "rule"
        if has_list and not is_summary:return INTENT_LIST,       0.90, "rule"
        if has_desc:                   return INTENT_DESCRIBE,   0.90, "rule"

        intent = cls._llm_classify(query)
        return intent, 0.60, "llm"

    @classmethod
    def _llm_classify(cls, query: str) -> str:
        prompt = (
            "Classify this query about police FIR records into exactly one:\n"
            "COUNT, DESCRIBE, LIST, ANALYTICAL, SIMILAR, LOOKUP\n\n"
            "Examples:\n"
            "  'how many accidents in 2019' -> COUNT\n"
            "  'describe assault cases in Udupi' -> DESCRIBE\n"
            "  'show all theft cases in Manipal' -> LIST\n"
            "  'which location has most crimes?' -> ANALYTICAL\n"
            "  'cases similar to night bike theft' -> SIMILAR\n\n"
            f"Query: {query}\nCategory:"
        )
        try:
            raw = _sarvam_call(prompt, max_tokens=10).upper()
            for intent in [INTENT_COUNT, INTENT_DESCRIBE, INTENT_HYBRID,
                           INTENT_LIST, INTENT_LOOKUP, INTENT_ANALYTICAL, INTENT_SIMILAR]:
                if intent in raw:
                    return intent
        except Exception:
            pass
        return INTENT_DESCRIBE


# ===========================================================================
# SQLiteManager — uses ONLY actual columns
# ===========================================================================

class SQLiteManager:

    def __init__(self, db_path: str = SQLITE_DB_PATH):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # Build WHERE clause from entity dict
    def _build_where(
        self,
        year: Optional[int]       = None,
        month: Optional[int]      = None,
        crime: Optional[str]      = None,
        location: Optional[str]   = None,
        day_label: Optional[str]  = None,
        time_label: Optional[str] = None,
        is_weekend: bool          = False,
    ) -> Tuple[str, List]:
        clauses: List[str] = []
        params: List[Any]  = []

        if year:
            clauses.append("CAST(year AS INTEGER) = ?")
            params.append(year)
        if month:
            clauses.append("CAST(month AS INTEGER) = ?")
            params.append(month)
        if crime:
            clauses.append("(crime_type_normalized LIKE ? OR crime_type LIKE ?)")
            params.extend([f"%{crime}%", f"%{crime}%"])
        if location:
            clauses.append("(location_normalized LIKE ? OR location LIKE ?)")
            params.extend([f"%{location}%", f"%{location}%"])
        if day_label:
            clauses.append("day_of_week_label = ?")
            params.append(day_label)
        elif is_weekend:
            clauses.append("is_weekend = 'True'")
        if time_label:
            clauses.append("time_of_day_label = ?")
            params.append(time_label)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        return where, params

    def _db_kw(self, e: Dict) -> Dict:
        return dict(
            year=e.get("year"), month=e.get("month"),
            crime=e.get("crime"), location=e.get("location"),
            day_label=e.get("day_label"), time_label=e.get("time_label"),
            is_weekend=e.get("is_weekend", False),
        )

    # Columns safe to select (all actually exist)
    _SAFE_COLS = (
        "record_id, crime_type, crime_type_normalized, location, location_normalized, "
        "day_of_week_label, time_of_day_label, day, month, month_name, year, "
        "is_weekend, date_key, crime_description"
    )

    def count(self, **kw) -> Tuple[int, str, List]:
        where, params = self._build_where(**kw)
        sql = f"SELECT COUNT(*) FROM fir_cases {where}"
        try:
            conn   = self._connect()
            result = conn.execute(sql, params).fetchone()[0]
            conn.close()
            return int(result), sql, params
        except Exception as exc:
            return -1, sql, params

    def retrieve(self, limit: int = MAX_LIST_ROWS, **kw) -> Tuple[List[Dict], str, List]:
        where, params = self._build_where(**kw)
        sql = f"SELECT {self._SAFE_COLS} FROM fir_cases {where} LIMIT ?"
        params_with_limit = params + [limit]
        try:
            conn = self._connect()
            rows = [dict(r) for r in conn.execute(sql, params_with_limit).fetchall()]
            conn.close()
            return rows, sql, params
        except Exception as exc:
            return [], sql, params

    def group_analytics(
        self, group_col: str, top_n: int = TOP_K_ANALYTICS, **kw
    ) -> Tuple[List[Tuple[str, int]], str, List]:
        allowed = {"location","crime_type","year","month","day_of_week_label","time_of_day_label"}
        if group_col not in allowed:
            return [], "", []
        where, params = self._build_where(**kw)
        null_guard = f"{group_col} IS NOT NULL AND TRIM(CAST({group_col} AS TEXT)) != ''"
        full_where = f"{where} AND {null_guard}" if where else f"WHERE {null_guard}"
        sql = (
            f"SELECT {group_col}, COUNT(*) AS cnt FROM fir_cases {full_where} "
            f"GROUP BY {group_col} ORDER BY cnt DESC LIMIT ?"
        )
        params_with_limit = params + [top_n]
        try:
            conn = self._connect()
            rows = conn.execute(sql, params_with_limit).fetchall()
            conn.close()
            return [(str(r[0]), int(r[1])) for r in rows], sql, params
        except Exception as exc:
            return [], sql, params

    def keyword_search(self, keyword: str, limit: int = 8) -> List[Dict]:
        sql = (
            f"SELECT {self._SAFE_COLS} FROM fir_cases "
            "WHERE crime_description LIKE ? LIMIT ?"
        )
        try:
            conn = self._connect()
            rows = [dict(r) for r in conn.execute(sql, [f"%{keyword}%", limit]).fetchall()]
            conn.close()
            return rows
        except Exception:
            return []

    def fetch_by_record_ids(self, record_ids: List[str], limit: int = 5) -> List[Dict]:
        if not record_ids:
            return []
        placeholders = ",".join("?" * len(record_ids))
        sql = f"SELECT {self._SAFE_COLS} FROM fir_cases WHERE record_id IN ({placeholders}) LIMIT ?"
        try:
            conn = self._connect()
            rows = [dict(r) for r in conn.execute(sql, record_ids + [limit]).fetchall()]
            conn.close()
            return rows
        except Exception:
            return []

    def fetch_by_record_id(self, record_id: str) -> Optional[Dict]:
        sql = f"SELECT {self._SAFE_COLS} FROM fir_cases WHERE record_id = ? LIMIT 1"
        try:
            conn = self._connect()
            row  = conn.execute(sql, [record_id]).fetchone()
            conn.close()
            return dict(row) if row else None
        except Exception:
            return None


# ===========================================================================
# ChromaVectorStore — uses actual metadata fields
# ===========================================================================

@st.cache_resource(show_spinner=False)
def _load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def _load_chroma_collection():
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=CHROMA_COLLECTION)   # no embedding_function — embeddings already stored
    return collection


class ChromaVectorStore:

    @staticmethod
    def search(query: str, top_k: int = TOP_K_CHROMA) -> Tuple[List[str], List[Dict], List[str]]:
        """Returns (docs, metadatas, record_ids)."""
        try:
            model      = _load_embedding_model()
            query_emb  = model.encode(["query: " + query]).tolist()
            col        = _load_chroma_collection()
            results    = col.query(
                query_embeddings=query_emb,
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            docs    = results.get("documents", [[]])[0]
            metas   = results.get("metadatas", [[]])[0]
            rids = []
            for m in metas:
                if m and "record_id" in m and m["record_id"] not in rids:
                    rids.append(str(m["record_id"]))
            return docs, metas, rids
        except Exception as exc:
            return [], [], []

    @staticmethod
    def parse_doc(doc: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        field_order = [
            "Crime Type","Location","Day of Week","Time of Day",
            "Day","Month","Year","Description"
        ]
        for i, key in enumerate(field_order):
            next_keys = field_order[i + 1:]
            lookahead = "|".join(re.escape(k + ":") for k in next_keys)
            pattern   = (
                rf"{re.escape(key)}:\s*(.*?)(?=\s+(?:{lookahead})|$)"
                if lookahead else rf"{re.escape(key)}:\s*(.*)"
            )
            m = re.search(pattern, doc, re.DOTALL)
            if m:
                out[key.lower().replace(" ", "_")] = m.group(1).strip()
        return out


# ===========================================================================
# Sarvam LLM helper
# ===========================================================================

def _strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = text[:text.index("<think>")]
    return text.strip()

def _sarvam_call(prompt: str, max_tokens: int = 800) -> str:
    if not SARVAM_API_KEY:
        return "[SARVAM_API_KEY not configured]"
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "model": SARVAM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.15,
    }
    resp = requests.post(SARVAM_API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    return _strip_think(raw)


def _detect_lang(text: str) -> str:
    for ch in text:
        if 0x0C80 <= ord(ch) <= 0x0CFF:
            return "kannada"
    return "english"


# ===========================================================================
# EvidenceBuilder
# ===========================================================================

class EvidenceBuilder:

    @staticmethod
    def from_rows(rows: List[Dict]) -> Dict:
        if not rows:
            return {}
        loc_ctr  = Counter(r.get("location_normalized") or r.get("location","") for r in rows)
        type_ctr = Counter(r.get("crime_type_normalized") or r.get("crime_type","") for r in rows)
        snippets = []
        for r in rows[:MAX_EVIDENCE_ROWS]:
            desc = (r.get("crime_description") or "")[:MAX_SNIPPET_LEN].strip()
            if desc:
                ct   = r.get("crime_type_normalized") or r.get("crime_type","")
                loc  = r.get("location_normalized")   or r.get("location","")
                mo   = r.get("month_name") or r.get("month","")
                yr   = r.get("year","")
                snippets.append(f"[{ct} | {loc} | {mo} {yr}]: {desc}")
        return dict(
            total=len(rows),
            top_locations=[l for l,_ in loc_ctr.most_common(3)  if l],
            top_crimes   =[c for c,_ in type_ctr.most_common(3) if c],
            snippets=snippets,
        )

    @staticmethod
    def to_prompt_text(ev: Dict) -> str:
        if not ev:
            return "ಯಾವುದೇ ಪ್ರಕರಣಗಳು ಕಂಡುಬಂದಿಲ್ಲ."
        lines = [f"ಪ್ರಕರಣಗಳ ಸಂಖ್ಯೆ: {ev['total']}"]
        if ev.get("top_locations"):
            lines.append("ಮುಖ್ಯ ಸ್ಥಳಗಳು: " + ", ".join(ev["top_locations"]))
        if ev.get("top_crimes"):
            lines.append("ಅಪರಾಧ ವಿಧಗಳು: " + ", ".join(ev["top_crimes"]))
        lines.extend(ev.get("snippets", []))
        return "\n".join(lines)


# ===========================================================================
# ResponseFormatter
# ===========================================================================

class ResponseFormatter:
    _STOP = [
        "</s>","<s>","[/INST]","ಪ್ರಶ್ನೆ:","Query:","Question:",
        "Examples:","Note:","ಗಮನಿಸಿ:","User:","Human:","Assistant:",
    ]

    @classmethod
    def clean(cls, text: str) -> str:
        text = text.strip()
        for pat in cls._STOP:
            if pat in text:
                text = text[:text.index(pat)].strip()
        text  = re.sub(r"[ \t]+", " ", text)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        text  = "\n\n".join(lines)
        if text and not text.endswith((".", "।", "ದೆ", "ವೆ")):
            text += "."
        return text


# ===========================================================================
# FIRAgent — main orchestrator
# ===========================================================================

def _is_off_topic(query: str) -> bool:
    ql = query.lower()
    if any(kw in ql or kw in query for kw in FIR_KEYWORDS):
        return False
    if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", query):
        return True
    return False


class FIRAgent:

    def __init__(self):
        self.db   = SQLiteManager()
        self.fmt  = ResponseFormatter()

    # ── Public entry ─────────────────────────────────────────────────────────

    def process(self, query: str) -> Dict:
        t0 = time.time()

        trace: Dict[str, Any] = dict(
            intent="", route="", answer_mode=AMODE_STATIC,
            classification_method="", confidence="",
            filters=[], rows_used=0, sql_count=None,
            llm_used=False, fallback_used=False,
            record_ids=[], chunk_count=0,
            sql="", sql_params=[], structured_preview="",
            chunk_preview="", error="", latency=0.0,
        )

        try:
            # Guard
            if _is_off_topic(query):
                trace["intent"] = INTENT_OFF_TOPIC
                trace["route"]  = "NONE"
                trace["latency"] = round(time.time()-t0, 2)
                return dict(answer="This query does not appear to be related to FIR crime records.", trace=trace)

            entities               = QueryParser.parse(query)
            style                  = ResponseStyleDetector.detect(query)
            intent, conf, method   = IntentClassifier.classify(query)
            lang                   = _detect_lang(query)

            trace["intent"]                = intent
            trace["classification_method"] = method
            trace["confidence"]            = f"{conf:.0%}"
            trace["filters"]               = self._filter_parts(entities)

            # Early-exit guards
            if entities.get("unknown_location"):
                trace["answer_mode"] = AMODE_STATIC
                trace["latency"] = round(time.time()-t0, 2)
                return dict(
                    answer=(
                        "ಈ ಸ್ಥಳವು ಉಡುಪಿ ಜಿಲ್ಲೆಯ ಡೇಟಾಸೆಟ್‌ನಲ್ಲಿ ಕಂಡುಬಂದಿಲ್ಲ. "
                        "*(Location not found. This system covers Udupi district FIRs only.)*"
                    ),
                    trace=trace,
                )
            if entities.get("unknown_crime") and intent in (
                INTENT_COUNT, INTENT_LIST, INTENT_DESCRIBE, INTENT_HYBRID, INTENT_ANALYTICAL
            ):
                trace["answer_mode"] = AMODE_STATIC
                trace["latency"] = round(time.time()-t0, 2)
                return dict(
                    answer=(
                        "ಈ ಅಪರಾಧ ವಿಧವು ಗುರುತಿಸಲಾಗಿಲ್ಲ. "
                        "ಬೆಂಬಲಿತ: ಹಲ್ಲೆ, ಕಳವು, ಅಪಘಾತ, ಜುಗಾರಿ, ಗಾಂಜಾ, ಕಾಣೆ, ಅಸ್ವಾಭಾವಿಕ, ವಂಚನೆ.\n\n"
                        "*(Crime type not recognised. Supported: assault, theft, accident, gambling, "
                        "drugs, missing, unnatural death, fraud.)*"
                    ),
                    trace=trace,
                )

            dispatch = {
                INTENT_COUNT:      self._handle_count,
                INTENT_DESCRIBE:   self._handle_describe,
                INTENT_HYBRID:     self._handle_hybrid,
                INTENT_LIST:       self._handle_list,
                INTENT_LOOKUP:     self._handle_lookup,
                INTENT_ANALYTICAL: self._handle_analytical,
                INTENT_SIMILAR:    self._handle_similar,
            }
            handler = dispatch.get(intent, self._handle_describe)
            answer  = handler(query, entities, style, lang, trace)

        except Exception:
            trace["error"] = traceback.format_exc()[:600]
            answer = "An error occurred while processing your query."

        trace["latency"] = round(time.time()-t0, 2)
        return dict(answer=answer, trace=trace)

    # ── COUNT ─────────────────────────────────────────────────────────────────

    def _handle_count(self, query, e, style, lang, trace):
        cnt, sql, params = self.db.count(**self.db._db_kw(e))
        trace.update(
            route="SQL", answer_mode=AMODE_SQL_COUNT,
            sql=sql, sql_params=params, sql_count=cnt
        )
        if cnt < 0:
            return "ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ."
        fl = self._filter_label(e)
        crime_part = f"{e['crime']} ಸಂಬಂಧಿತ " if e.get("crime") else ""
        return f"{fl}{crime_part}ಒಟ್ಟು **{cnt}** ಅಪರಾಧ ಪ್ರಕರಣಗಳು ದಾಖಲಾಗಿವೆ.\n\n*ಮೂಲ: SQL*"

    # ── DESCRIBE ──────────────────────────────────────────────────────────────

    def _handle_describe(self, query, e, style, lang, trace):
        want_summary = ResponseStyleDetector.is_summary_style(style)

        if e["has_filters"]:
            rows, sql, params = self.db.retrieve(limit=MAX_EVIDENCE_ROWS, **self.db._db_kw(e))
            trace.update(sql=sql, sql_params=params)

            if rows:
                trace.update(route="SQL", rows_used=len(rows),
                             structured_preview=json.dumps(rows[:2], ensure_ascii=False, default=str)[:500])
                ev = EvidenceBuilder.from_rows(rows)
                if want_summary:
                    return self._sql_summary(query, e, ev, lang, trace)
                else:
                    trace["answer_mode"] = AMODE_SQL_DIRECT
                    return self._render_cards(ev, e, "SQL")

            # Structured query, no SQL rows
            docs, metas, rids = ChromaVectorStore.search(query)
            trace.update(fallback_used=True, record_ids=rids, chunk_count=len(docs),
                         chunk_preview=" | ".join(d[:80] for d in docs[:2]))
            no_match = (
                "ನಿಮ್ಮ ಮಾನದಂಡಕ್ಕೆ ಹೊಂದಿಕೆಯಾಗುವ ದಾಖಲೆಗಳು ಕಂಡುಬಂದಿಲ್ಲ.\n\n"
                "*(No exact structured records matched your filters.)*"
            )
            if not docs:
                trace["answer_mode"] = AMODE_STATIC
                return no_match
            trace.update(route="SQL+VEC", answer_mode=AMODE_SEM_RELATED, rows_used=len(docs))
            return no_match + "\n\n" + self._semantic_cards(docs, "approximate semantic matches")

        # No filters — semantic summary
        docs, metas, rids = ChromaVectorStore.search(query)
        trace.update(
            route="VEC", fallback_used=True,
            record_ids=rids, chunk_count=len(docs),
            chunk_preview=" | ".join(d[:80] for d in docs[:2]),
        )
        if not docs:
            return "ಸಂಬಂಧಿತ FIR ದಾಖಲೆಗಳು ಕಂಡುಬಂದಿಲ್ಲ."
        trace.update(answer_mode=AMODE_SEM_SUMMARY, llm_used=True, rows_used=len(docs))
        context = "\n\n".join(docs[:4])
        lang_instr = "Respond in Kannada." if lang=="kannada" else "Respond in English."
        prompt = (
            f"Using ONLY the FIR records below, answer: \"{query}\"\n"
            "Do not invent names, events or details not in the records.\n"
            f"{lang_instr}\nWrite 4-5 sentences.\n\nRecords:\n{context}\n\nAnswer:"
        )
        try:
            resp = _sarvam_call(prompt)
            return self.fmt.clean(resp) + "\n\n*ಮೂಲ: Semantic search*"
        except Exception as exc:
            return f"[LLM unavailable] Semantic results: {context[:300]}"

    # ── HYBRID ────────────────────────────────────────────────────────────────

    def _handle_hybrid(self, query, e, style, lang, trace):
        cnt, sql_c, params_c = self.db.count(**self.db._db_kw(e))
        rows, sql_r, params_r = self.db.retrieve(limit=MAX_EVIDENCE_ROWS, **self.db._db_kw(e))
        trace.update(
            route="SQL", sql=sql_r, sql_params=params_r, sql_count=cnt,
        )
        fl         = self._filter_label(e)
        crime_part = f"{e['crime']} ಸಂಬಂಧಿತ " if e.get("crime") else ""
        count_line = (f"**ಸಂಖ್ಯೆ:** {fl}{crime_part}ಒಟ್ಟು {cnt} ಪ್ರಕರಣಗಳು." if cnt >= 0 else "")

        if rows:
            trace.update(rows_used=len(rows),
                         structured_preview=json.dumps(rows[:2], ensure_ascii=False, default=str)[:500])
            ev = EvidenceBuilder.from_rows(rows)

            if style == STYLE_LIST:
                trace["answer_mode"] = AMODE_SQL_HYBRID
                lines = ([count_line, ""] if count_line else [])
                lines.append(f"**ವಿವರ ({ev['total']} ಪ್ರಕರಣಗಳು):**\n")
                for snip in ev["snippets"]:
                    lines.append(f"> {snip}")
                lines.append("*ಮೂಲ: SQL*")
                return "\n\n".join(lines)

            # Summary mode (default for HYBRID)
            trace.update(answer_mode=AMODE_SQL_HYBRID, llm_used=True)
            ev_txt = EvidenceBuilder.to_prompt_text(ev)
            fl2    = self._filter_label(e) or "ಆಯ್ದ "
            lang_instr = "Respond in Kannada." if lang=="kannada" else "Respond in English."
            prompt = (
                "Use ONLY the data below. Do not invent names or details.\n"
                f"{lang_instr} Write 4-5 precise sentences.\n\n"
                f"{count_line}\n\n{fl2}ಪ್ರಕರಣಗಳ ಮಾಹಿತಿ:\n{ev_txt}\n\nವಿವರಣೆ:"
            )
            try:
                resp = _sarvam_call(prompt)
                return (
                    f"{count_line}\n\n**ವಿವರ:** {self.fmt.clean(resp)}"
                    f"\n\n*ಮೂಲ: SQL ({ev['total']} ಪ್ರಕರಣಗಳು)*"
                )
            except Exception:
                return count_line + "\n\n" + self._render_cards(ev, e, "SQL")

        # No SQL rows — semantic fallback
        docs, metas, rids = ChromaVectorStore.search(query)
        trace.update(
            route="SQL+VEC", fallback_used=True,
            record_ids=rids, chunk_count=len(docs),
            chunk_preview=" | ".join(d[:80] for d in docs[:2]),
        )
        if not docs:
            return count_line or "ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ."
        if e["has_filters"]:
            trace["answer_mode"] = AMODE_SEM_RELATED
            no_match = "ನಿರ್ದಿಷ್ಟ ದಾಖಲೆಗಳು ಕಂಡುಬಂದಿಲ್ಲ. ಸಂಬಂಧಿತ ಪ್ರಕರಣಗಳನ್ನು ತೋರಿಸಲಾಗಿದೆ."
            lines = ([count_line, no_match] if count_line else [no_match])
            lines.append(self._semantic_cards(docs[:3], "approximate matches"))
            return "\n\n".join(lines)
        # Broad query
        trace.update(answer_mode=AMODE_SEM_SUMMARY, llm_used=True, rows_used=len(docs))
        context = "\n\n".join(docs)
        lang_instr = "Respond in Kannada." if lang=="kannada" else "Respond in English."
        prompt = (
            f"{count_line}\n\nUsing ONLY these FIR records, write a 4-5 sentence answer to: \"{query}\"\n"
            f"Do not invent details.\n{lang_instr}\n\nRecords:\n{context}\n\nAnswer:"
        )
        try:
            resp = _sarvam_call(prompt)
            return f"{count_line}\n\n**ವಿವರ:** {self.fmt.clean(resp)}\n\n*ಮೂಲ: SQL + Semantic*"
        except Exception:
            return count_line + "\n\n" + self._semantic_cards(docs[:3], "semantic fallback")

    # ── LIST ──────────────────────────────────────────────────────────────────

    def _handle_list(self, query, e, style, lang, trace):
        rows, sql, params = self.db.retrieve(limit=MAX_LIST_ROWS, **self.db._db_kw(e))
        trace.update(
            route="SQL", answer_mode=AMODE_SQL_DIRECT,
            sql=sql, sql_params=params, rows_used=len(rows),
            structured_preview=json.dumps(rows[:2], ensure_ascii=False, default=str)[:500],
        )
        if not rows:
            return "ಸಂಬಂಧಿತ FIR ಪ್ರಕರಣಗಳು ಕಂಡುಬಂದಿಲ್ಲ."
        fl    = self._filter_label(e)
        lines = [f"**{fl}{len(rows)} ಪ್ರಕರಣಗಳು:**\n"]
        for i, r in enumerate(rows, 1):
            ct   = r.get("crime_type_normalized") or r.get("crime_type","—")
            loc  = r.get("location_normalized")   or r.get("location","—")
            mo   = r.get("month_name") or r.get("month","")
            yr   = r.get("year","")
            rid  = r.get("record_id","")
            desc = (r.get("crime_description") or "")[:130].strip()
            lines.append(f"**{i}.** `{rid}` · {ct} · {loc} · {mo} {yr}\n> {desc}...")
        lines.append("*ಮೂಲ: SQL*")
        return "\n\n".join(lines)

    # ── LOOKUP ────────────────────────────────────────────────────────────────

    def _handle_lookup(self, query, e, style, lang, trace):
        # Exact record_id lookup
        if e.get("record_id"):
            row = self.db.fetch_by_record_id(e["record_id"])
            trace.update(route="SQL", answer_mode=AMODE_SQL_DIRECT, rows_used=1 if row else 0)
            if row:
                trace["structured_preview"] = json.dumps(row, ensure_ascii=False, default=str)[:500]
                return self._format_single_row(row)

        # Filter-based lookup
        if e["has_filters"]:
            rows, sql, params = self.db.retrieve(limit=5, **self.db._db_kw(e))
            trace.update(route="SQL", sql=sql, sql_params=params, rows_used=len(rows))
            if rows:
                trace.update(
                    answer_mode=AMODE_SQL_DIRECT,
                    structured_preview=json.dumps(rows[:2], ensure_ascii=False, default=str)[:500],
                )
                ev = EvidenceBuilder.from_rows(rows)
                return self._render_cards(ev, e, "SQL")

        # Keyword search
        tokens = [t for t in re.split(r"\s+", query) if len(t) > 3][:4]
        for tok in tokens:
            rows = self.db.keyword_search(tok, limit=5)
            if rows:
                trace.update(route="SQL", answer_mode=AMODE_SQL_DIRECT, rows_used=len(rows))
                ev = EvidenceBuilder.from_rows(rows)
                return self._render_cards(ev, e, "SQL keyword search")

        # Semantic fallback
        docs, metas, rids = ChromaVectorStore.search(query)
        trace.update(
            route="VEC", fallback_used=True,
            record_ids=rids, chunk_count=len(docs),
            chunk_preview=" | ".join(d[:80] for d in docs[:2]),
            answer_mode=AMODE_SEM_RELATED, rows_used=len(docs),
        )
        if not docs:
            return "ಈ ಪ್ರಶ್ನೆಗೆ ಸಂಬಂಧಿತ FIR ಪ್ರಕರಣಗಳು ಕಂಡುಬಂದಿಲ್ಲ."
        return "**ಸಂಭಾವ್ಯ ಹೊಂದಾಣಿಕೆ (Semantic):**\n\n" + self._semantic_cards(docs[:4], "")

    # ── ANALYTICAL ────────────────────────────────────────────────────────────

    def _handle_analytical(self, query, e, style, lang, trace):
        ql = query.lower()
        if any(kw in ql for kw in ("ಸ್ಥಳ","location","place","where","ಎಲ್ಲಿ","hotspot")):
            grp, label = "location", "ಸ್ಥಳ"
        elif any(kw in ql for kw in ("ವರ್ಷ","year","annual")):
            grp, label = "year", "ವರ್ಷ"
        elif any(kw in ql for kw in ("ತಿಂಗಳ","month","monthly")):
            grp, label = "month", "ತಿಂಗಳ"
        elif any(kw in ql for kw in ("crime type","ಅಪರಾಧ ವಿಧ","which crime","ಯಾವ ಅಪರಾಧ")):
            grp, label = "crime_type", "ಅಪರಾಧ ವಿಧ"
        elif any(kw in ql for kw in ("time","ಸಮಯ","time_of_day")):
            grp, label = "time_of_day_label", "ಸಮಯ"
        elif any(kw in ql for kw in ("day","ದಿನ","weekday")):
            grp, label = "day_of_week_label", "ದಿನ"
        else:
            grp, label = "location", "ಸ್ಥಳ"

        loc_arg  = e["location"] if grp != "location" else None
        kw_args  = dict(
            year=e["year"], month=e["month"], crime=e["crime"],
            location=loc_arg, day_label=e["day_label"],
            time_label=e["time_label"], is_weekend=e["is_weekend"],
        )
        results, sql, params = self.db.group_analytics(
            group_col=grp, top_n=TOP_K_ANALYTICS, **kw_args
        )
        trace.update(
            route="SQL", answer_mode=AMODE_SQL_ANALYTICS,
            sql=sql, sql_params=params, rows_used=len(results),
        )
        if not results:
            return "ವಿಶ್ಲೇಷಣೆ ಮಾಡಲು ಸಾಕಷ್ಟು ಮಾಹಿತಿ ಇಲ್ಲ."

        fl     = self._filter_label(e)
        header = f"**{fl}{label}ವಾರು ವಿಶ್ಲೇಷಣೆ (ಅಗ್ರ {len(results)}):**\n"
        body   = []
        for rank, (val, cnt) in enumerate(results, 1):
            if grp == "month" and str(val).isdigit():
                val = MONTH_NUM_KN.get(int(val), val)
            body.append(f"{rank}. **{val}** — {cnt} ಪ್ರಕರಣಗಳು")
        return header + "\n".join(body) + f"\n\n*ಮೂಲ: SQL GROUP BY {grp}*"

    # ── SIMILAR ───────────────────────────────────────────────────────────────

    def _handle_similar(self, query, e, style, lang, trace):
        search_q = f"{e['crime']} {query}" if e.get("crime") else query
        docs, metas, rids = ChromaVectorStore.search(search_q, top_k=6)
        trace.update(
            route="VEC", fallback_used=True,
            record_ids=rids, chunk_count=len(docs),
            chunk_preview=" | ".join(d[:80] for d in docs[:2]),
            answer_mode=AMODE_SEM_RELATED, rows_used=len(docs),
        )
        if not docs:
            return "ಇದೇ ರೀತಿಯ FIR ಪ್ರಕರಣಗಳು ಕಂಡುಬಂದಿಲ್ಲ."

        # Enrich with SQL rows
        sql_rows = self.db.fetch_by_record_ids(rids[:5]) if rids else []
        if sql_rows:
            trace["structured_preview"] = json.dumps(sql_rows[:2], ensure_ascii=False, default=str)[:500]

        return "**ಸಮಾನ FIR ಪ್ರಕರಣಗಳು:**\n\n" + self._semantic_cards(docs[:5], "ChromaDB")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _sql_summary(self, query, e, ev, lang, trace):
        trace.update(answer_mode=AMODE_SQL_SUMMARY, llm_used=True)
        ev_txt = EvidenceBuilder.to_prompt_text(ev)
        fl     = self._filter_label(e) or "ಆಯ್ದ "
        lang_instr = "Respond in Kannada." if lang=="kannada" else "Respond in English."
        prompt = (
            "Use ONLY the data below. Do not invent names, events or outside details.\n"
            f"{lang_instr}\n"
            "Write 4-5 precise sentences covering location, crime type and context.\n\n"
            f"{fl}FIR ಪ್ರಕರಣಗಳ ಮಾಹಿತಿ:\n{ev_txt}\n\nಸಾರಾಂಶ:"
        )
        try:
            resp = _sarvam_call(prompt)
            return self.fmt.clean(resp) + f"\n\n*ಮೂಲ: SQL ({ev['total']} ಪ್ರಕರಣಗಳು)*"
        except Exception as exc:
            trace["llm_used"] = False
            return self._render_cards(ev, e, "SQL")

    def _render_cards(self, ev: Dict, e: Dict, source: str) -> str:
        fl    = self._filter_label(e) or ""
        lines = [f"**{fl}{ev['total']} ಪ್ರಕರಣಗಳು ಕಂಡುಬಂದಿವೆ:**\n"]
        for snip in ev.get("snippets", []):
            lines.append(f"> {snip}")
        lines.append(f"*ಮೂಲ: {source}*")
        return "\n\n".join(lines)

    def _semantic_cards(self, docs: List[str], label: str) -> str:
        lines = []
        if label:
            lines.append(f"**{label}:**\n")
        for i, doc in enumerate(docs, 1):
            p    = ChromaVectorStore.parse_doc(doc)
            ct   = p.get("crime_type",   "—")
            loc  = p.get("location",     "—")
            dt   = p.get("day",          "")
            mo   = p.get("month",        "")
            yr   = p.get("year",         "")
            desc = p.get("description",  doc[:200])[:200]
            lines.append(f"**{i}.** {ct} · {loc} · {dt}/{mo}/{yr}\n> {desc}...")
        lines.append("*ಮೂಲ: Semantic search (ChromaDB)*")
        return "\n\n".join(lines)

    def _format_single_row(self, r: Dict) -> str:
        ct   = r.get("crime_type_normalized") or r.get("crime_type","—")
        loc  = r.get("location_normalized")   or r.get("location","—")
        mo   = r.get("month_name") or r.get("month","")
        yr   = r.get("year","")
        dow  = r.get("day_of_week_label","")
        tod  = r.get("time_of_day_label","")
        rid  = r.get("record_id","")
        desc = r.get("crime_description","")
        return (
            f"**Record:** `{rid}`\n\n"
            f"**Crime:** {ct}  \n**Location:** {loc}  \n"
            f"**Date:** {mo} {yr}  \n**Day:** {dow}  \n**Time:** {tod}\n\n"
            f"**Description:** {desc}"
        )

    @staticmethod
    def _filter_parts(e: Dict) -> List[str]:
        parts = []
        if e.get("year"):       parts.append(str(e["year"]))
        if e.get("month_name"): parts.append(e["month_name"])
        if e.get("crime"):      parts.append(e["crime"])
        if e.get("location"):   parts.append(e["location"])
        if e.get("time_label"): parts.append(e["time_label"])
        if e.get("is_weekend") and not e.get("day_label"):
            parts.append("Weekend")
        elif e.get("day_label"):
            parts.append(e["day_label"])
        return parts

    def _filter_label(self, e: Dict) -> str:
        parts = self._filter_parts(e)
        return (" · ".join(parts) + " — ") if parts else ""


# ===========================================================================
# Streamlit Chat UI
# ===========================================================================

# Header
st.markdown("""
<div class="fir-header">
    <h1>FIR INTELLIGENCE SYSTEM</h1>
    <p>Udupi District  &middot;  Hybrid SQL + Semantic  &middot;  Sarvam LLM  &middot;  v3</p>
</div>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of (role, message, trace | None)

if "query_history" not in st.session_state:
    st.session_state.query_history = []  # list of user query strings

if "agent" not in st.session_state:
    with st.spinner("Initialising..."):
        _load_chroma_collection()        # warm up embedding model
        st.session_state.agent = FIRAgent()

# ── Sidebar: previous questions ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("**Recent queries**")
    if st.session_state.query_history:
        for i, q in enumerate(reversed(st.session_state.query_history[-15:]), 1):
            st.markdown(f"`{i}.` {q[:60]}{'…' if len(q)>60 else ''}")
    else:
        st.caption("No queries yet.")

# ── Empty state ───────────────────────────────────────────────────────────────

if not st.session_state.chat_history:
    st.markdown("""
    <div class="empty-hint">
        <p>
            ಅಪರಾಧ ಪ್ರಕರಣಗಳ ಬಗ್ಗೆ ಕೇಳಿ:<br/>
            <code>2019ರಲ್ಲಿ ಹಲ್ಲೆ ಪ್ರಕರಣಗಳು ಎಷ್ಟು?</code><br/>
            <code>ಉಡುಪಿಯಲ್ಲಿ ಕಳವು ಪ್ರಕರಣಗಳ ಸಾಮಾನ್ಯ ಸ್ವರೂಪ ವಿವರಿಸಿ</code><br/>
            <code>ಯಾವ ಸ್ಥಳದಲ್ಲಿ ಹೆಚ್ಚು ಅಪಘಾತ ಪ್ರಕರಣಗಳು?</code><br/>
            <code>How many accident cases in 2018?</code><br/>
            <code>Show me theft cases in Manipal 2020</code>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Trace render helpers ─────────────────────────────────────────────────────

def _render_trace_bar(trace: Dict):
    filters_str = ", ".join(trace.get("filters", [])) or "none"
    llm_f   = "yes" if trace.get("llm_used")     else "no"
    fall_f  = "yes" if trace.get("fallback_used") else "no"
    conf    = trace.get("confidence", "—")
    st.markdown(
        f'<div class="trace-bar">'
        f'intent: <span>{trace.get("intent","—")}</span> &nbsp;·&nbsp; '
        f'route: <span>{trace.get("route","—")}</span> &nbsp;·&nbsp; '
        f'mode: <span>{trace.get("answer_mode","—")}</span> &nbsp;·&nbsp; '
        f'classify: <span>{trace.get("classification_method","—")}</span> &nbsp;·&nbsp; '
        f'conf: <span>{conf}</span> &nbsp;·&nbsp; '
        f'rows: <span>{trace.get("rows_used",0)}</span> &nbsp;·&nbsp; '
        f'llm: <span>{llm_f}</span> &nbsp;·&nbsp; '
        f'fallback: <span>{fall_f}</span> &nbsp;·&nbsp; '
        f'filters: <span>{filters_str}</span> &nbsp;·&nbsp; '
        f'latency: <span>{trace.get("latency","—")}s</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_trace_expander(trace: Dict):
    with st.expander("Backend details"):

        if trace.get("filters"):
            st.markdown('<div class="sec-label">Extracted filters</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="detail-block">{json.dumps(trace["filters"], ensure_ascii=False)}</div>',
                unsafe_allow_html=True,
            )

        if trace.get("sql"):
            st.markdown('<div class="sec-label">SQL executed</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="detail-block">{trace["sql"]}\n\nparams: {trace["sql_params"]}</div>',
                unsafe_allow_html=True,
            )

        if trace.get("sql_count") is not None:
            st.markdown('<div class="sec-label">SQL count</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="detail-block">{trace["sql_count"]}</div>',
                unsafe_allow_html=True,
            )

        if trace.get("structured_preview"):
            st.markdown('<div class="sec-label">Structured row preview</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="detail-block">{trace["structured_preview"][:600]}</div>',
                unsafe_allow_html=True,
            )

        if trace.get("record_ids"):
            st.markdown('<div class="sec-label">Retrieved record IDs</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="detail-block">{trace["record_ids"][:10]}</div>',
                unsafe_allow_html=True,
            )

        if trace.get("chunk_preview"):
            st.markdown('<div class="sec-label">Chunk preview</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="detail-block">{trace["chunk_preview"][:300]}</div>',
                unsafe_allow_html=True,
            )

        if trace.get("error"):
            st.markdown('<div class="sec-label">Error</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="detail-block err">{trace["error"]}</div>',
                unsafe_allow_html=True,
            )


# ── Render existing chat history ─────────────────────────────────────────────

for role, msg, trace in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)
        if trace and role == "assistant":
            _render_trace_bar(trace)
            _render_trace_expander(trace)


# ── Chat input & response ─────────────────────────────────────────────────────

query = st.chat_input("ನಿಮ್ಮ ಪ್ರಶ್ನೆ ಟೈಪ್ ಮಾಡಿ / Type your question...")

if query and query.strip():
    user_q = query.strip()

    # Store question
    st.session_state.query_history.append(user_q)
    st.session_state.chat_history.append(("user", user_q, None))

    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ..."):
            result = st.session_state.agent.process(user_q)

        answer = result["answer"]
        trace  = result["trace"]

        st.markdown(answer)
        _render_trace_bar(trace)
        _render_trace_expander(trace)

    st.session_state.chat_history.append(("assistant", answer, trace))

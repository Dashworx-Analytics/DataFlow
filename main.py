import csv
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ========= CONFIG (edit these two) =========
INPUT_FILE = "dataset.xlsx"
OUTPUT_DIR = "clean_output"
# ==========================================

# general settings
DAYFIRST_HINT = True          # True if dates are usually DD/MM/YYYY
THRESH_NUMERIC = 0.88         # (kept, but numeric inference now requires 100% numeric)
THRESH_DATE = 0.65            # share of parsable rows to accept date
MAX_ROWS_SAMPLE = 20000       # speed cap per column for inference
DECIMAL_CHAR = "."            # set "," if decimals use comma
CURRENCY_CHARS = "£$€¥₹"
NBSP = "\u00A0"
EXCEL_EPOCH = datetime(1899, 12, 30)
EXCEL_DATE_MIN = 20000
EXCEL_DATE_MAX = 60000
BOOL_TRUE = {"true", "t", "yes", "y", "1"}
BOOL_FALSE = {"false", "f", "no", "n", "0"}

NA_MAP = {
    "": np.nan,
    "nan": np.nan,
    "none": np.nan,
    "null": np.nan,
    "N/A": np.nan,
    "n/a": np.nan,
    "-": np.nan,
    "\u2014": np.nan  # em dash as unicode escape
}

COMMON_DATE_PATTERNS = [
    "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y",
    "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
    "%d.%m.%Y", "%Y.%m.%d",
    "%d-%b-%Y", "%d %b %Y", "%b %d %Y",
    "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S"
]

def norm_header(s: str) -> str:
    s = str(s).replace(NBSP, " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

# UPDATED: robust header policy (BQ-safe)
def simple_header(s: str) -> str:
    """
    - Trim extra spaces
    - Keep only alphanumerics; split on any non-alphanum
    - If 1 token -> keep original token casing
    - If 2+ tokens -> snake_case in lowercase
    """
    s_norm = norm_header(s)
    if s_norm == "":
        return s_norm
    tokens = re.findall(r"[A-Za-z0-9]+", s_norm)
    if not tokens:
        return "col"
    if len(tokens) == 1:
        return tokens[0]
    return "_".join(t.lower() for t in tokens)

def strip_cell(x):
    if isinstance(x, str):
        # normalize curly quotes; trim; map common NA strings to NaN
        x = x.replace("“", "\"").replace("”", "\"").replace("’", "'").replace(NBSP, " ").strip()
        return NA_MAP.get(x, x)
    return x

def sample_series(s: pd.Series, n: int) -> pd.Series:
    return s.sample(n, random_state=42) if len(s) > n else s

def detect_id_like(s: pd.Series) -> bool:
    x = s.dropna().astype(str)
    if x.empty:
        return False
    letter_ratio = x.str.contains(r"[A-Za-z]").mean()
    lead0_ratio = x.str.match(r"^0+\d+$").mean()
    unique_ratio = x.nunique(dropna=True) / max(len(x), 1)
    allowed = x.str.match(r"^[A-Za-z0-9\-\_/\.]+$").mean()
    return (letter_ratio > 0.15 or lead0_ratio > 0.05) and unique_ratio > 0.4 and allowed > 0.7

def normalize_numeric_text(s: pd.Series) -> pd.Series:
    # Convert to string and trim whitespace, but preserve NaN
    x = s.copy()
    mask = x.notna()
    
    if mask.any():
        # Only process non-null values
        x_str = x.loc[mask].astype(str).str.strip()
        
        # Replace NaN string representations and empty strings back to actual NaN
        x_str = x_str.replace("nan", np.nan)
        x_str = x_str.replace("None", np.nan)
        x_str = x_str.replace("", np.nan)
        
        # Map NA strings to NaN before normalization
        for na_str, na_val in NA_MAP.items():
            x_str = x_str.replace(na_str, na_val)
        
        # Apply normalizations using vectorized string operations
        # Only apply to non-null values
        x_str = x_str.str.replace(r"^\((.*)\)$", r"-\1", regex=True)   # (123) -> -123
        x_str = x_str.str.replace(r"^(.+)-$", r"-\1", regex=True)      # 123- -> -123
        x_str = x_str.str.replace(f"[{re.escape(CURRENCY_CHARS)}{NBSP} ]", "", regex=True)
        x_str = x_str.str.replace(",", "", regex=False)                # drop thousands comma
        if DECIMAL_CHAR != ".":
            x_str = x_str.str.replace(DECIMAL_CHAR, ".", regex=False)
        x_str = x_str.str.replace("%", "", regex=False)
        
        # Update the original series with processed values
        x.loc[mask] = x_str
    
    return x

def try_numeric(s: pd.Series):
    x = normalize_numeric_text(s)
    num = pd.to_numeric(x, errors="coerce")
    # Calculate ratio: successfully converted / original non-null values
    original_non_null = s.notna().sum()
    if original_non_null == 0:
        return num, 0.0
    successfully_converted = num.notna().sum()
    ratio = successfully_converted / original_non_null
    return num, ratio

def try_parse_date_patterns(s: pd.Series):
    x = s.astype(str)
    best = None
    best_ratio = -1.0
    for fmt in COMMON_DATE_PATTERNS:
        dt = pd.to_datetime(x, format=fmt, errors="coerce")
        ratio = dt.notna().mean()
        if ratio > best_ratio:
            best_ratio = ratio
            best = dt
            if best_ratio == 1.0:
                break
    return best, best_ratio

def try_parse_date_direction(s: pd.Series, dayfirst: bool):
    dt = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, infer_datetime_format=False, utc=False)
    return dt, dt.notna().mean()

def try_parse_excel_serial(s: pd.Series):
    num, ratio = try_numeric(s)
    if ratio < 0.7:
        return None, 0.0
    nonnull = num.dropna()
    if nonnull.empty:
        return None, 0.0
    frac = np.modf(nonnull.values)[0]
    intlike_ratio = float(np.mean(frac == 0)) if len(frac) else 0.0
    within_range = ((nonnull >= EXCEL_DATE_MIN) & (nonnull <= EXCEL_DATE_MAX)).mean()
    if intlike_ratio > 0.9 and within_range > 0.9:
        dt = nonnull.apply(lambda v: EXCEL_EPOCH + timedelta(days=int(v)))
        out = pd.Series(pd.NaT, index=num.index, dtype="datetime64[ns]")
        out.loc[nonnull.index] = pd.to_datetime(dt.values)
        return out, out.notna().mean()
    return None, 0.0

def has_text_values(s: pd.Series) -> bool:
    """
    Check if column contains any text/letter values (not just numbers or boolean).
    Returns True if column contains letters/words, indicating it should be STRING.
    """
    x = s.dropna().astype(str).str.strip()
    if x.empty:
        return False
    
    # Check if any values contain letters (A-Z, a-z)
    has_letters = x.str.contains(r"[A-Za-z]", na=False)
    if has_letters.any():
        # If we have any values with letters, it's text
        # But exclude boolean values to avoid false positives
        x_lower = x.str.lower()
        boolean_values = BOOL_TRUE | BOOL_FALSE
        non_boolean_text = has_letters & ~x_lower.isin(boolean_values)
        
        # If there are any non-boolean text values, it's definitely STRING
        if non_boolean_text.any():
            text_vals = x[non_boolean_text]
            # If any text value is longer than 1 char, it's a word (not a code)
            if (text_vals.str.len() > 1).any():
                return True
            # Even single char codes indicate it's text
            if text_vals.nunique() >= 1:
                return True
    
    return False

def has_non_boolean_text(s: pd.Series) -> bool:
    """
    Check if column contains text/letters that are NOT boolean-like.
    Returns True if column contains words or text patterns that indicate it's not a boolean column.
    This prevents columns with text values like "Chevron", "Herringbone" from being misclassified as BOOL.
    """
    x = s.dropna().astype(str).str.strip()
    if x.empty:
        return False
    
    # Convert to lowercase for comparison
    x_lower = x.str.lower()
    boolean_values = BOOL_TRUE | BOOL_FALSE
    
    # Check if any values contain letters
    has_letters = x.str.contains(r"[A-Za-z]", na=False)
    if not has_letters.any():
        return False  # No letters at all, can't be text
    
    # Find values that have letters but are NOT in boolean sets
    # These are text values like "Chevron", "Herringbone", etc.
    non_boolean_with_letters = has_letters & ~x_lower.isin(boolean_values)
    
    if non_boolean_with_letters.any():
        # Get the actual non-boolean text values
        text_values = x[non_boolean_with_letters]
        
        # If ANY of these are longer than 1 character, it's definitely text (not boolean)
        # This catches "Chevron", "Herringbone", etc.
        # This is the main check - any word-like text value means it's not boolean
        if (text_values.str.len() > 1).any():
            return True
        
        # Even for single characters, if we have ANY non-boolean letter value,
        # it's likely a code/identifier column, not boolean
        # This catches cases like "a", "b", "c" codes mixed with "0"
        if text_values.nunique() >= 1:
            return True
    
    return False

def detect_boolean(s: pd.Series) -> bool:
    x = s.dropna().astype(str).str.strip().str.lower()
    if x.empty:
        return False
    return x.isin(BOOL_TRUE | BOOL_FALSE).mean() > 0.9

def coerce_boolean(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.lower()
    return x.map(lambda v: True if v in BOOL_TRUE else (False if v in BOOL_FALSE else np.nan)).astype("boolean")

def infer_column(s: pd.Series, name: str):
    """
    Robust inference for client-safe CSV -> BigQuery Autodetect:
    - Check for text/letters first to avoid misclassifying text columns as boolean
    - Try boolean only if column doesn't contain non-boolean text
    - Try numeric with reasonable threshold (90%+) to catch numeric data
    - Try date detection to catch dates (including those with letters)
    - Then check id-like patterns
    - Default to STRING
    """
    s = s.apply(strip_cell)

    # Check for text/letters FIRST - if column contains text values, it should be STRING
    # This prevents columns with text like "Chevron", "Herringbone" from being classified as numeric
    if has_text_values(s):
        # Column contains text values, classify as STRING (don't check numeric or boolean)
        return s.astype(str).str.strip(), "STRING", None

    # Check for non-boolean text to prevent misclassification as boolean
    # If column contains text values like "Chevron", "Herringbone", etc., skip boolean detection
    if has_non_boolean_text(s):
        # Column contains text that's not boolean-like, skip boolean check
        pass
    else:
        # Only check boolean if column doesn't contain non-boolean text
        if detect_boolean(s):
            return coerce_boolean(s), "BOOL", None

    # numeric check with reasonable threshold (90%+ numeric)
    # This catches numeric data before it gets misclassified as STRING
    num, num_ratio = try_numeric(s)
    # Also check if majority of non-null values are numeric (more lenient)
    non_null_count = s.notna().sum()
    if non_null_count > 0:
        numeric_count = num.notna().sum()
        # Use the higher of: ratio of converted values, or absolute count threshold
        if num_ratio >= 0.90 or (numeric_count >= 3 and num_ratio >= 0.80):  # 90%+ or 80%+ with at least 3 values
            nonnull = num.dropna()
            if len(nonnull) > 0:
                # Check if all non-null values are integers
                if np.all(np.modf(nonnull.values)[0] == 0):
                    return num.astype("Int64"), "INT64", None
                return num.astype(float), "FLOAT64", None

    # dates - CHECK AFTER numeric to avoid misclassifying numeric dates
    ss = sample_series(s, MAX_ROWS_SAMPLE)

    excel_dt, excel_ratio = try_parse_excel_serial(ss)
    if excel_ratio >= THRESH_DATE:
        num, _ = try_numeric(s)
        nonnull = num.dropna()
        out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
        idx = nonnull.index
        out.loc[idx] = [EXCEL_EPOCH + timedelta(days=int(v)) for v in nonnull.loc[idx]]
        return out, "DATE", "%Y-%m-%d"

    pat_dt, pat_ratio = try_parse_date_patterns(ss)
    d1, r1 = try_parse_date_direction(ss, dayfirst=DAYFIRST_HINT)
    d2, r2 = try_parse_date_direction(ss, dayfirst=not DAYFIRST_HINT)
    candidates = [(pat_dt, pat_ratio), (d1, r1), (d2, r2)]
    best_dt, best_ratio = max(candidates, key=lambda t: t[1])

    if best_ratio >= THRESH_DATE:
        if best_dt is pat_dt:
            full_dt, _ = try_parse_date_patterns(s)
        elif best_dt is d1:
            full_dt, _ = try_parse_date_direction(s, dayfirst=DAYFIRST_HINT)
        else:
            full_dt, _ = try_parse_date_direction(s, dayfirst=not DAYFIRST_HINT)

        nonnull = full_dt.dropna()
        bq_type = "TIMESTAMP"
        date_fmt = "%Y-%m-%d %H:%M:%S"
        if not nonnull.empty and (nonnull.dt.time == datetime.min.time()).all():
            bq_type = "DATE"
            date_fmt = "%Y-%m-%d"
        return full_dt, bq_type, date_fmt

    # id-like -> STRING
    if detect_id_like(s):
        return s.astype(str).str.strip(), "STRING", None

    # default STRING
    return s.astype(str).str.strip(), "STRING", None

def coerce_column_to_type(s: pd.Series, target_type: str, date_format: str | None = None):
    """
    Coerce a column to a given BigQuery type using existing helpers.
    Used when the user edits the schema and we want to enforce it.
    
    Args:
        s: Series to coerce
        target_type: Target BigQuery type
        date_format: Optional date format string (e.g., "%Y-%m-%d", "%m/%d/%Y") for DATE/TIMESTAMP types
    """
    s = s.apply(strip_cell)

    t = target_type.upper()

    if t == "STRING":
        return s.astype(str).str.strip(), "STRING", None

    if t == "BOOL":
        return coerce_boolean(s), "BOOL", None

    if t in {"INT64", "FLOAT64"}:
        num, _ = try_numeric(s)
        if t == "INT64":
            return num.astype("Int64"), "INT64", None
        return num.astype(float), "FLOAT64", None

    if t in {"DATE", "TIMESTAMP"}:
        # If user provided a date format, try to parse with it first
        if date_format:
            try:
                # Try parsing with the specified format
                parsed = pd.to_datetime(s, format=date_format, errors='coerce')
                if parsed.notna().mean() >= THRESH_DATE:
                    # Successfully parsed enough values with this format
                    fmt = date_format
                    return parsed, t, fmt
            except (ValueError, TypeError):
                # Format parsing failed, fall back to auto-detection
                pass
        
        # Auto-detect date format if user format didn't work or wasn't provided
        ss = sample_series(s, MAX_ROWS_SAMPLE)
        excel_dt, excel_ratio = try_parse_excel_serial(ss)
        pat_dt, pat_ratio = try_parse_date_patterns(ss)
        d1, r1 = try_parse_date_direction(ss, dayfirst=DAYFIRST_HINT)
        d2, r2 = try_parse_date_direction(ss, dayfirst=not DAYFIRST_HINT)
        candidates = [(excel_dt, excel_ratio), (pat_dt, pat_ratio), (d1, r1), (d2, r2)]
        best_dt, _ = max(candidates, key=lambda x: x[1])

        if best_dt is excel_dt:
            num, _ = try_numeric(s)
            nonnull = num.dropna()
            out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
            idx = nonnull.index
            out.loc[idx] = [EXCEL_EPOCH + timedelta(days=int(v)) for v in nonnull.loc[idx]]
            full_dt = out
        elif best_dt is pat_dt:
            full_dt, _ = try_parse_date_patterns(s)
        elif best_dt is d1:
            full_dt, _ = try_parse_date_direction(s, dayfirst=DAYFIRST_HINT)
        else:
            full_dt, _ = try_parse_date_direction(s, dayfirst=not DAYFIRST_HINT)

        # Use user-provided format if available, otherwise use default (%d/%m/%Y format)
        fmt = date_format if date_format else ("%d/%m/%Y" if t == "DATE" else "%d/%m/%Y %H:%M:%S")
        return full_dt, t, fmt

    # fallback
    return s.astype(str).str.strip(), "STRING", None

def _map_bq_type_for_schema(t: str) -> str:
    """
    Normalize internal type names to BigQuery schema type names
    used in JSON schema and text schema.
    """
    t_upper = t.upper()
    if t_upper in {"INT64", "INTEGER"}:
        return "INTEGER"
    if t_upper in {"FLOAT64", "FLOAT"}:
        return "FLOAT"
    if t_upper in {"BOOL", "BOOLEAN"}:
        return "BOOLEAN"
    # STRING, DATE, TIMESTAMP already match what BigQuery expects
    return t_upper

def bq_schema_from_df(df: pd.DataFrame, date_fmt_map: dict) -> list:
    schema = []
    for col in df.columns:
        if col in date_fmt_map and date_fmt_map[col] == "%Y-%m-%d":
            bq = "DATE"
        else:
            dtype = df[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                bq = "INT64"
            elif pd.api.types.is_float_dtype(dtype):
                bq = "FLOAT64"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                bq = "TIMESTAMP"
            elif dtype.name == "boolean":
                bq = "BOOL"
            else:
                bq = "STRING"
        schema.append({
            "description": "",
            "name": col,
            "type": _map_bq_type_for_schema(bq),
            "mode": "NULLABLE"
        })
    return schema

def format_dates_for_csv(df: pd.DataFrame, date_fmt_map: dict) -> pd.DataFrame:
    """
    Format date columns according to the specified format in date_fmt_map.
    Uses the user-selected format directly without BigQuery compliance restrictions.
    """
    out = df.copy()
    for col, fmt in date_fmt_map.items():
        if col in out.columns and pd.api.types.is_datetime64_any_dtype(out[col]):
            # Format dates using the specified format from date_fmt_map
            out[col] = out[col].dt.strftime(fmt)
    return out

def reorder_for_bq_autodetect(df: pd.DataFrame, bq_type_map: dict) -> pd.DataFrame:
    """
    Move rows with letters in STRING columns to the top so BigQuery Autodetect
    reliably infers STRING for those columns. Deterministic, no data changes.
    """
    string_cols = [c for c, t in bq_type_map.items() if t == "STRING" and c in df.columns]
    if not string_cols:
        return df
    has_letters = pd.Series(False, index=df.index)
    for c in string_cols:
        s = df[c].astype(str)
        has_letters = has_letters | s.str.contains(r"[A-Za-z]", na=False)
    out = df.copy()
    out["__letters__"] = has_letters.astype(int)
    out = out.sort_values("__letters__", ascending=False).drop(columns="__letters__")
    return out

def write_clean_csv(df: pd.DataFrame, path: Path):
    # Write as-is and tell pandas how to render missing values
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        quotechar='"',
        doublequote=True,
        lineterminator="\n",
        encoding="utf-8",
        na_rep=""
    )

def find_unbalanced_quote_lines(path: Path):
    bad = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for i, line in enumerate(f, 1):
            if line.count('"') % 2:
                bad.append(i)
    return bad

def write_bq_text_schema(bq_type_map: dict, path: Path):
    """
    Write BigQuery schema in text form suitable for 'Edit as text' box.
    One field per line: name:TYPE,MODE  (MODE is usually NULLABLE)
    """
    with open(path, "w", encoding="utf-8") as f:
        for col, typ in bq_type_map.items():
            f.write(f"{col}:{_map_bq_type_for_schema(typ)},NULLABLE\n")

def process_sheet(sheet_name: str, df_raw: pd.DataFrame, out_dir: Path, override_types: dict | None = None, override_date_formats: dict | None = None):
    # Header cleanup
    df_raw.columns = [simple_header(c) for c in df_raw.columns]

    # Cell cleanup
    df_raw = df_raw.applymap(strip_cell)

    # Inference or coercion
    typed = {}
    date_fmt_map = {}
    bq_type_map = {}

    for col in df_raw.columns:
        if override_types is not None and col in override_types:
            # Get user-specified date format for this column if available
            user_date_format = None
            if override_date_formats and col in override_date_formats:
                user_date_format = override_date_formats[col]
            ser, bq_type, date_fmt = coerce_column_to_type(df_raw[col], override_types[col], date_format=user_date_format)
        else:
            ser, bq_type, date_fmt = infer_column(df_raw[col], col)

        typed[col] = ser
        bq_type_map[col] = bq_type
        if date_fmt:
            date_fmt_map[col] = date_fmt
        elif bq_type == "TIMESTAMP":
            date_fmt_map[col] = "%d/%m/%Y %H:%M:%S"

    df_clean = pd.DataFrame(typed)
    df_to_write = format_dates_for_csv(df_clean, date_fmt_map)

    # Ensure autodetect sees STRING columns as text early
    df_to_write = reorder_for_bq_autodetect(df_to_write, bq_type_map)

    # Outputs
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", sheet_name).strip("_") or "Sheet"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{safe}.csv"
    schema_path = out_dir / f"{safe}_bq_schema.json"
    schema_text_path = out_dir / f"{safe}_bq_schema.txt"
    summary_path = out_dir / f"{safe}_summary.txt"

    write_clean_csv(df_to_write, csv_path)

    bad_lines = find_unbalanced_quote_lines(csv_path)
    if bad_lines:
        print(f"Warning: {len(bad_lines)} line(s) have unbalanced quotes. Example lines: {bad_lines[:5]}")

    schema = bq_schema_from_df(df_clean, date_fmt_map)
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    write_bq_text_schema(bq_type_map, schema_text_path)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Sheet: {sheet_name}\n")
        for col in df_clean.columns:
            f.write(f"- {col}: {bq_type_map[col]}\n")

    print(f"OK: {csv_path.name}, {schema_path.name}, {schema_text_path.name}, {summary_path.name}")

def process_xlsx(xlsx_path: Path, out_dir: Path):
    sheets = pd.read_excel(
        xlsx_path,
        sheet_name=None,
        dtype=str,
        keep_default_na=False,
        engine="openpyxl"
    )
    for name, df in sheets.items():
        process_sheet(name, df, out_dir)

def process_csv(csv_path: Path, out_dir: Path):
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, engine="python", on_bad_lines="skip")
    process_sheet(csv_path.stem, df, out_dir)

def main():
    in_path = Path(INPUT_FILE)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    suf = in_path.suffix.lower()
    if suf in {".xlsx", ".xlsm", ".xls"}:
        process_xlsx(in_path, out_dir)
    elif suf == ".csv":
        process_csv(in_path, out_dir)
    else:
        raise ValueError("Unsupported file type. Provide .xlsx or .csv")

if __name__ == "__main__":
    main()
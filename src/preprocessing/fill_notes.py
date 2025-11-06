# -*- coding: utf-8 -*-
"""
preprocessing.fill_notes
Purpose: Enrich raw JSON episode documents with DRG predictions and auxiliary metadata.
Stage: preprocessing — reads raw JSONs, queries Oracle for historical DRG and patient dates,
writes formatted JSONs (enriched) and a CSV log, and moves files without a confident pre-DRG.
Example: from preprocessing.fill_notes import update_json_files; use functions in a pipeline
See: CLI/runner that calls this module for batch processing.

New / common additional classification labels and metadata fields beyond `drg_target`:
- drg_target_lib: human-readable label for drg_target (already used in output).
- predrg_min, predrg_max: candidate pre-DRG lists (min = earliest, max = chosen closest to ref).
- predrg_date_min, predrg_date_max: corresponding dates.
- predecoded_tags: (example) extra predicted tags from a classifier (severity, urgency).
- severity_score: numeric severity label produced by other models.
- predicted_icd: predicted ICD code for the episode.
- confidence: float confidence for predicted labels.
Note: update this list to match the real dataset schema; fields above are common examples.
"""

import os
import json
from tqdm import tqdm

try:  # pragma: no cover - import guard exercised in runtime environments
    import cx_Oracle  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback executed when cx_Oracle missing
    try:  # pragma: no cover - secondary fallback for thin client
        import oracledb as cx_Oracle  # type: ignore  # python-oracledb exposes a compatible API
    except ModuleNotFoundError:  # pragma: no cover - stub mode handles missing driver lazily
        cx_Oracle = None  # type: ignore[assignment]

import pandas as pd
import csv
import re
from datetime import datetime, date
import shutil


DEFAULT_INSTANT_CLIENT_DIR = os.environ.get("ORACLE_INSTANT_CLIENT_PATH", r"C:\gic\oracle\instantclient_19_6")


def ensure_instant_client_on_path(path_hint: str | None) -> None:
    """Prepend the Oracle Instant Client directory to PATH when provided."""

    if not path_hint:
        return
    normalized = os.path.normpath(path_hint)
    current = os.environ.get("PATH", "")
    segments = [segment.strip() for segment in current.split(";") if segment.strip()]
    if normalized in segments:
        return
    updated = ";".join([normalized] + segments) if segments else normalized
    os.environ["PATH"] = updated

# =============== CONFIG DEBUG SQL =================
DEBUG_PRINT_SQL = False
DEBUG_SQL_PATH = None  # None => auto = output_directory/oracle_query_<ts>.sql
# ==================================================

# =========================
# Utils dates
# =========================
def parse_date_any(s: str):
    """
    Parse a date/time string into a datetime.date when possible.

    Accepts several common formats and falls back to pandas.to_datetime with dayfirst=True.
    Returns:
        datetime.date or None
    Raises:
        None — failures return None (caller should handle missing dates).
    """
    if not isinstance(s, str) or not s.strip():
        return None
    s = s.strip().replace("\u00a0", " ")
    # try common explicit formats first (fast)
    for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    # last resort: pandas flexible parser (handles many edge cases)
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.notna(dt):
            return dt.date()
    except Exception:
        pass
    return None

def to_iso(d: date) -> str:
    """
    Convert a date object to ISO YYYY-MM-DD or return empty string for invalid input.
    """
    return d.strftime("%Y-%m-%d") if isinstance(d, date) else ""

def closest_drg_to_date(predrg_list, predrg_date_list, ref_date_iso: str):
    """
    Return (drg_code, date_iso) from predrg_list closest to ref_date_iso.

    - If multiple entries tie on distance and share the same date, the LAST occurrence is chosen.
    - Inputs:
        predrg_list: list of DRG codes (strings) or []
        predrg_date_list: list of date strings (various formats) or []
        ref_date_iso: reference date as 'YYYY-MM-DD' (string); if falsy, returns two empty strings.
    - Returns:
        (best_drg_code_or_empty, best_date_iso_or_empty)
    """
    if not predrg_list or not predrg_date_list or not ref_date_iso:
        return "", ""
    try:
        ref_d = datetime.strptime(ref_date_iso, "%Y-%m-%d").date()
    except Exception:
        return "", ""

    best_diff = None
    best_date = None
    best_drg, best_iso = "", ""

    # iterate in supplied order; tie-breaker rule picks latest occurrence if same date
    for drg, dstr in zip(predrg_list, predrg_date_list):
        d = parse_date_any(dstr)
        if not d:
            continue
        diff = abs((d - ref_d).days)

        # closer date -> replace
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_date = d
            best_drg = (drg or "").strip()
            best_iso = to_iso(d)
        # equal distance and same date -> pick this later occurrence
        elif diff == best_diff and best_date is not None and d == best_date:
            best_drg = (drg or "").strip()
            best_iso = to_iso(d)

    return best_drg, best_iso

# =========================
# JSON → collecte EDS/doc
# =========================
def collect_eds_ids(input_directory):
    """
    Scan input_directory for JSON files and collect EDS and document identifiers.

    Returns:
        (eds_ids_list, json_files_list)
        - eds_ids_list: list of EDS ids (as strings) to query Oracle with.
        - json_files_list: list of tuples (path, filename, eds_id_str, doc_id_str) for processing.
    Notes:
        - Robust to file read errors: prints an error and continues.
    """
    eds_ids = []
    json_files = []
    for filename in os.listdir(input_directory):
        if not filename.lower().endswith(".json"):
            continue
        path = os.path.join(input_directory, filename)
        if os.path.isdir(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # tolerate multiple possible keys for EDS and doc identifiers
            eds_id = data.get("eds_id") or data.get("eds") or data.get("eds_doc") or ""
            doc_id = data.get("doc_id") or data.get("document_id") or data.get("doc_version_id") or ""
            if eds_id not in (None, ""):
                eds_ids.append(eds_id)
                json_files.append((path, filename, str(eds_id), str(doc_id)))
        except Exception as e:
            # lightweight debug info — callers can redirect stdout to logs
            print(f"[Erreur lecture JSON] {filename}: {e}")
    return eds_ids, json_files

# =========================
# Helpers SQL Developer
# =========================
def _oracle_quote(s: str) -> str:
    """Quote a string for inclusion in a SQL literal (simple single-quote escaping)."""
    return "'" + str(s).replace("'", "''") + "'"

def build_sql_for_sqldeveloper(eds_ids, eds_are_numeric: bool) -> str:
    """
    Build a SQL string suitable for pasting into SQL Developer for debugging or reproduction.

    Args:
        eds_ids: iterable of EDS ids (as strings).
        eds_are_numeric: True if EDS values are numeric (no quotes), False otherwise.
    Returns:
        A SQL string with the collection literal expanded for SQL Developer.
    """
    seen, eds_clean = set(), []
    for v in eds_ids:
        sv = str(v).strip()
        if sv and sv not in seen:
            seen.add(sv)
            eds_clean.append(sv)

    if eds_are_numeric:
        coll_ctor = f"SYS.ODCINUMBERLIST({','.join(eds_clean)})"
    else:
        coll_ctor = f"SYS.ODCIVARCHAR2LIST({','.join(_oracle_quote(x) for x in eds_clean)})"

    sql = r"""
WITH ranked AS (
  SELECT
      t.EDS,
      t.DRG_PREVU_CODE,
      t.DT_GROUPAGE,
      ROW_NUMBER() OVER (
        PARTITION BY t.EDS
        ORDER BY t.DT_GROUPAGE ASC, t.DRG_PREVU_CODE
      ) AS rn
  FROM (
      SELECT
          d.EDS,
          UPPER(TRIM(REGEXP_REPLACE(DRG_PREVU_CODE, '\s+', ' '))) AS DRG_PREVU_CODE,
          TRUNC(DT_GROUPAGE) AS DT_GROUPAGE
      FROM COD_DRG d 
      JOIN TABLE({coll_ctor}) p
        ON d.EDS = p.COLUMN_VALUE
      WHERE d.FLAG_PRECODAGE = 1 
      GROUP BY
          d.EDS,
          UPPER(TRIM(REGEXP_REPLACE(DRG_PREVU_CODE, '\s+', ' '))),
          TRUNC(DT_GROUPAGE)
  ) t
),
top10 AS (
  SELECT * FROM ranked WHERE rn <= 10
),
drg_list AS (
  SELECT
      t.EDS,
      LISTAGG(t.DRG_PREVU_CODE, '|') WITHIN GROUP (ORDER BY t.DT_GROUPAGE ASC) AS DRG_LIST,
      LISTAGG(TO_CHAR(t.DT_GROUPAGE, 'YYYY-MM-DD'), '|') WITHIN GROUP (ORDER BY t.DT_GROUPAGE ASC) AS DATE_LIST
  FROM top10 t
  GROUP BY t.EDS
)
SELECT
    a.NOPTN,
    a.EDS,
    COALESCE(a.DRG_OPA_CODE, a.DRG_DPI_CODE_INI) AS DRG_OPA_CODE,
    CASE 
        WHEN a.DRG_OPA_CODE IS NULL THEN a.DRG_DPI_LIB_INI
        ELSE a.DRG_OPA_LIB
    END AS DRG_OPA_LIB,
    a.SERMED_ID_LST,
    a.SERMED_MNM_LST,
    a.DT_DEB_SEJ,
    a.DT_FIN_SEJ,
    a.AGE_ANNEE,
    b.DRG_LIST       AS PRE_DRG_LIST,
    b.DATE_LIST      AS PRE_DRG_DATE_LIST
FROM dm_episode_drg a
LEFT JOIN drg_list b ON a.EDS = b.EDS
WHERE a.EDS IN (SELECT COLUMN_VALUE FROM TABLE({coll_ctor}))
  AND (
        ( b.DRG_LIST IS NOT NULL
          AND NOT REGEXP_LIKE(
                UPPER(TRIM(REGEXP_SUBSTR(b.DRG_LIST, '^[^|]+'))),
                '^(TR|TP|9)'
              )
        )
        OR (
             b.DRG_LIST IS NULL
             AND NOT REGEXP_LIKE(
                   UPPER(TRIM(COALESCE(a.DRG_OPA_CODE, a.DRG_DPI_CODE_INI)))),
                   '^(TR|TP|9)'
                 )
           )
      )
""".strip()
    return sql.format(coll_ctor=coll_ctor)

# =========================
# Oracle (chunked, with optional SQL print)
# =========================
SQL_MAIN = r"""
WITH ranked AS (
  SELECT
      t.EDS,
      t.DRG_PREVU_CODE,
      t.DT_GROUPAGE,
      ROW_NUMBER() OVER (
        PARTITION BY t.EDS
        ORDER BY t.DT_GROUPAGE ASC, t.DRG_PREVU_CODE
      ) AS rn
  FROM (
      SELECT
          d.EDS,
          UPPER(TRIM(REGEXP_REPLACE(DRG_PREVU_CODE, '\\s+', ' '))) AS DRG_PREVU_CODE,
          TRUNC(DT_GROUPAGE) AS DT_GROUPAGE
      FROM COD_DRG d 
      JOIN TABLE(CAST(:eds AS {coll_type})) p
        ON d.EDS = p.COLUMN_VALUE
      WHERE d.FLAG_PRECODAGE = 1 
      GROUP BY
          d.EDS,
          UPPER(TRIM(REGEXP_REPLACE(DRG_PREVU_CODE, '\\s+', ' ')))),
          TRUNC(DT_GROUPAGE)
  ) t
),
top10 AS (
  SELECT * FROM ranked WHERE rn <= 10
),
drg_list AS (
  SELECT
      t.EDS,
      LISTAGG(t.DRG_PREVU_CODE, '|') WITHIN GROUP (ORDER BY t.DT_GROUPAGE ASC) AS DRG_LIST,
      LISTAGG(TO_CHAR(t.DT_GROUPAGE, 'YYYY-MM-DD'), '|') WITHIN GROUP (ORDER BY t.DT_GROUPAGE ASC) AS DATE_LIST
  FROM top10 t
  GROUP BY t.EDS
)
SELECT
    a.NOPTN,
    a.EDS,
    COALESCE(a.DRG_OPA_CODE, a.DRG_DPI_CODE_INI) AS DRG_OPA_CODE,
    CASE 
        WHEN a.DRG_OPA_CODE IS NULL THEN a.DRG_DPI_LIB_INI
        ELSE a.DRG_OPA_LIB
    END AS DRG_OPA_LIB,
    a.SERMED_ID_LST,
    a.SERMED_MNM_LST,
    a.DT_DEB_SEJ,
    a.DT_FIN_SEJ,
    a.AGE_ANNEE,
    b.DRG_LIST       AS PRE_DRG_LIST,
    b.DATE_LIST      AS PRE_DRG_DATE_LIST
FROM dm_episode_drg a
LEFT JOIN drg_list b ON a.EDS = b.EDS
WHERE a.EDS IN (SELECT COLUMN_VALUE FROM TABLE(CAST(:eds AS {coll_type})))
  AND (
        /* Cas 1 : DRG_LIST existe → ne garder que si le 1er élément ne commence pas par TR|TP|9 */
        ( b.DRG_LIST IS NOT NULL
          AND NOT REGEXP_LIKE(
                UPPER(TRIM(REGEXP_SUBSTR(b.DRG_LIST, '^[^|]+'))),
                '^(TR|TP|9)'
              )
        )
        /* Cas 2 : DRG_LIST est NULL → on retombe sur le filtre historique du code épisode */
        OR (
             b.DRG_LIST IS NULL
             AND NOT REGEXP_LIKE(
                   UPPER(TRIM(COALESCE(a.DRG_OPA_CODE, a.DRG_DPI_CODE_INI)))),
                   '^(TR|TP|9)'
                 )
           )
      )
"""

def fetch_oracle_data(
    eds_ids,
    oracle_dsn,
    oracle_user,
    oracle_password,
    *,
    batch_size: int = 5000,
    debug_print_sql: bool = False,
    debug_sql_path: str | None = None,
    instant_client_dir: str | None = None,
):
    """
    Query Oracle in batches to fetch episode-level DRG history and demographics.

    Args:
        eds_ids: iterable of EDS ids (strings).
        oracle_dsn, oracle_user, oracle_password: Oracle connection params.
        batch_size: number of EDS per collection batch (Oracle collection types).
    debug_print_sql: if True, prints a version of the SQL for SQL Developer use.
    debug_sql_path: optional path to write the debug SQL.
    instant_client_dir: optional filesystem path containing Oracle Instant Client DLLs.
    Returns:
        pandas.DataFrame with columns:
        ['NOPTN','EDS','DRG_OPA_CODE','DRG_OPA_LIB','SERMED_ID_LST','SERMED_MNM_LST',
         'DT_DEB_SEJ','DT_FIN_SEJ','AGE_ANNEE','PRE_DRG_LIST','PRE_DRG_DATE_LIST']
    Notes:
        - Uses Oracle collection types SYS.ODCINUMBERLIST or SYS.ODCIVARCHAR2LIST depending on EDS format.
        - All exceptions from cx_Oracle connect/execute will propagate to the caller.
    """
    if cx_Oracle is None:
        raise ModuleNotFoundError(
            "Oracle client driver not available. Install cx_Oracle/python-oracledb or enable stub mode via config."
        )

    ensure_instant_client_on_path(instant_client_dir or DEFAULT_INSTANT_CLIENT_DIR)
    eds_are_numeric = True
    for v in eds_ids:
        try:
            int(v)
        except Exception:
            eds_are_numeric = False
            break

    if debug_print_sql:
        sql_copy = build_sql_for_sqldeveloper(eds_ids, eds_are_numeric)
        print("\n================ SQL POUR SQL DEVELOPER ================\n")
        print(sql_copy)
        print("\n========================================================\n")
        if debug_sql_path:
            try:
                with open(debug_sql_path, "w", encoding="utf-8") as f:
                    f.write(sql_copy + "\n")
                print(f"[DEBUG] Requête SQL écrite dans: {debug_sql_path}")
            except Exception as e:
                print(f"[DEBUG] Impossible d'écrire {debug_sql_path}: {e}")

    # establish connection — let exceptions bubble up for caller to handle/log
    conn = cx_Oracle.connect(user=oracle_user, password=oracle_password, dsn=oracle_dsn)
    cur = conn.cursor()

    # choose appropriate Oracle collection type and conversion
    if eds_are_numeric:
        coll_name = "SYS.ODCINUMBERLIST"
        coll_type = conn.gettype(coll_name)
        to_item = int
    else:
        coll_name = "SYS.ODCIVARCHAR2LIST"
        coll_type = conn.gettype(coll_name)
        to_item = str

    sql = SQL_MAIN.format(coll_type=coll_name)

    results = []
    # chunked execution to avoid overly large collection literals
    for i in tqdm(range(0, len(eds_ids), batch_size), desc="Requêtes Oracle (chunks)"):
        chunk = eds_ids[i:i + batch_size]
        eds_obj = coll_type.newobject()
        for v in chunk:
            eds_obj.append(to_item(v))
        cur.execute(sql, eds=eds_obj)
        rows = cur.fetchall()
        results.extend(rows)

    cur.close()
    conn.close()

    cols = [
        'NOPTN','EDS','DRG_OPA_CODE','DRG_OPA_LIB','SERMED_ID_LST','SERMED_MNM_LST',
        'DT_DEB_SEJ','DT_FIN_SEJ','AGE_ANNEE','PRE_DRG_LIST','PRE_DRG_DATE_LIST'
    ]
    df = pd.DataFrame(results, columns=cols)

    # keep one row per EDS (first sorted by EDS), preserves deterministic behaviour
    if not df.empty:
        df = df.sort_values(['EDS']).drop_duplicates(subset=['EDS'], keep='first').reset_index(drop=True)

    return df

# =========================
# CSV meta (robust date updates)
# =========================
def load_metadata_csv(meta_csv_path: str) -> pd.DataFrame:
    """
    Load a metadata CSV that can provide update begin/end dates per (EDS, document).

    The loader is intentionally robust:
    - normalizes column names (removes punctuation/quotes/Unicode variants)
    - attempts to pick common aliases for eds/doc/update begin/update end
    - returns a DataFrame with canonical columns: eds_key, doc_key, update_begin_date_iso, update_end_date_iso, ...
    Args:
        meta_csv_path: path to CSV file
    Returns:
        pandas.DataFrame (empty if file missing or eds column absent)
    """
    if not meta_csv_path or not os.path.exists(meta_csv_path):
        print(f"[META] Fichier CSV introuvable: {meta_csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(
        meta_csv_path,
        encoding="utf-8-sig",
        sep=None,
        engine="python",
        dtype=str,
        keep_default_na=False
    )
    # sanitize column names
    df.columns = [c.replace("\ufeff", "").strip().strip('"').strip("“”«»").strip() for c in df.columns]

    def norm(name: str) -> str:
        name = (name or "").replace("\ufeff", "")
        name = name.strip().strip('"').strip("“”«»").strip()
        return re.sub(r"[^a-z0-9]+", "", name.lower())

    norm_map = {c: norm(c) for c in df.columns}
    alias_map = {
        "eds": "eds", "edsid": "eds", "eds_id": "eds", "edsdoc": "eds", "eds_doc": "eds",
        "neds": "eds", "edsdocid": "eds", "eds_doc_id": "eds",
        "_id": "doc", "id": "doc", "docversionid": "doc", "doc_version_id": "doc",
        "documentid": "doc", "document_id": "doc", "docid": "doc", "doc_id": "doc",
        "updatebegindate": "beg", "updatebegin": "beg", "updatebegindatetime": "beg",
        "updateenddate": "end", "updateend": "end", "updateenddatetime": "end",
    }

    # pick best-matching columns for EDS/doc/beg/end using normalization and aliases
    picked = {"eds": None, "doc": None, "beg": None, "end": None}
    for orig, nrm in norm_map.items():
        key = alias_map.get(nrm)
        if key and picked[key] is None:
            picked[key] = orig

    # dump column normalization mapping to help external devs inspect the mapping used
    dbg_dir = os.path.dirname(meta_csv_path) or "."
    dbg_path = os.path.join(dbg_dir, f"_columns_debug_{datetime.now():%Y%m%d_%H%M%S}.csv")
    pd.DataFrame({"original": list(df.columns), "normalized": [norm_map[c] for c in df.columns]}).to_csv(
        dbg_path, index=False, encoding="utf-8"
    )
    print(f"[META] Colonnes CSV dump → {dbg_path}")
    print(f"[META] Sélection colonnes → {picked}")

    if picked["eds"] is None:
        print("[META] Colonne EDS absente (ajouter un alias si besoin).")
        return pd.DataFrame()

    keep = [picked["eds"]]
    if picked["doc"]: keep.append(picked["doc"])
    if picked["beg"]: keep.append(picked["beg"])
    if picked["end"]: keep.append(picked["end"])
    meta = df[keep].copy()

    # sanitize values and compute ISO date fields (raw preserved)
    for c in keep:
        meta[c] = meta[c].astype(str).str.replace("\ufeff", "", regex=False).str.strip().str.strip('"').str.strip()

    meta["update_begin_date_iso"] = meta[picked["beg"]].apply(parse_date_any).apply(to_iso) if picked["beg"] else ""
    meta["update_end_date_iso"]   = meta[picked["end"]].apply(parse_date_any).apply(to_iso) if picked["end"] else ""
    meta["update_begin_date_raw"] = meta[picked["beg"]] if picked["beg"] else ""
    meta["update_end_date_raw"]   = meta[picked["end"]] if picked["end"] else ""

    meta["eds_key"] = meta[picked["eds"]].astype(str).str.strip()
    meta["doc_key"] = meta[picked["doc"]].astype(str).str.strip() if picked["doc"] else ""

    return meta

def lookup_meta_row(meta_df: pd.DataFrame, eds_id: str, doc_id: str):
    """
    Find the best metadata row for a given (eds_id, doc_id).

    Strategy:
      - if doc_id present, prefer exact (eds,doc) match (first row)
      - otherwise return the last row matching eds_id (most recent)
      - returns None if meta_df is empty or no match found
    """
    if meta_df.empty:
        return None
    eds_id = str(eds_id).strip()
    doc_id = str(doc_id).strip()
    if doc_id:
        sub = meta_df[(meta_df["eds_key"] == eds_id) & (meta_df["doc_key"] == doc_id)]
        if not sub.empty:
            return sub.iloc[0]
    sub = meta_df[meta_df["eds_key"] == eds_id]
    if not sub.empty:
        return sub.iloc[-1]
    return None

# =========================
# Update JSONs + log + move no-DRG files
# =========================
def update_json_files(json_files, df_oracle, meta_df, output_directory, log_csv_path):
    """
    Enrich JSON files using Oracle and CSV meta information, write outputs and a CSV log.

    Behavior:
      - For each JSON file found by collect_eds_ids, attempts to enrich with Oracle row (if present)
        and CSV meta dates (update_begin/update_end).
      - Chooses the pre-DRG closest to update_end_date_iso using closest_drg_to_date.
      - If no chosen pre-DRG (predrg_max empty) the formatted JSON (or original) is moved to a
        'no_drg_eds_files' subdirectory to mark it for manual review.
      - Writes a CSV log summarizing actions and statuses.

    Args:
        json_files: list of tuples (path, filename, eds_id, doc_id)
        df_oracle: DataFrame returned by fetch_oracle_data (may be empty)
        meta_df: DataFrame returned by load_metadata_csv (may be empty)
        output_directory: where to write formatted JSONs
        log_csv_path: path to write the CSV log (will be overwritten)
    """
    os.makedirs(output_directory, exist_ok=True)
    no_drg_dir = os.path.join(output_directory, "no_drg_eds_files")
    os.makedirs(no_drg_dir, exist_ok=True)

    log_rows = []

    # prepare oracle DF for efficient lookups by string EDS
    df_oracle = df_oracle.copy()
    stub_mode = bool(getattr(df_oracle, "attrs", {}).get("stub_mode"))
    if not df_oracle.empty:
        df_oracle["EDS_str"] = df_oracle["EDS"].astype(str)

    # process each JSON with progress bar
    for path, filename, eds_id, doc_id in tqdm(json_files, desc="Mise à jour JSON"):
        # minimal log row template — filled during processing
        row_data = {
            'file': filename,
            'eds_id': eds_id,
            'doc_id': doc_id,
            'drg_target': '',
            'predrg_max': '',
            'predrg_max_date': '',
            'update_begin_date_iso': '',
            'update_end_date_iso': '',
            'age_years': '',
            'status': '',
            'message': ''
        }

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # ---- Oracle enrichment (if available)
            pre_drg_list, pre_date_list = [], []
            age_from_oracle = None
            if not df_oracle.empty:
                match = df_oracle[df_oracle["EDS_str"] == str(eds_id)]
                if not match.empty:
                    # choose first row — df_oracle already de-duplicated upstream
                    row = match.iloc[0]
                    data["drg_target"]      = row.get("DRG_OPA_CODE") or ""
                    data["drg_target_lib"]  = row.get("DRG_OPA_LIB") or ""
                    data["sermed_id_lst"]   = row.get("SERMED_ID_LST") or ""
                    data["sermed_mnm_lst"]  = row.get("SERMED_MNM_LST") or ""
                    data["dt_deb_sej"]      = str(row.get("DT_DEB_SEJ") or "")
                    data["dt_fin_sej"]      = str(row.get("DT_FIN_SEJ") or "")

                    # AGE_ANNEE (if present) — use to fill missing age_years
                    if "AGE_ANNEE" in match.columns:
                        try:
                            age_from_oracle = int(row.get("AGE_ANNEE")) if row.get("AGE_ANNEE") not in (None, "", "None") else None
                        except Exception:
                            age_from_oracle = None

                    pre_drg_list  = (row.get("PRE_DRG_LIST") or "").split("|") if pd.notna(row.get("PRE_DRG_LIST")) else []
                    pre_date_list = (row.get("PRE_DRG_DATE_LIST") or "").split("|") if pd.notna(row.get("PRE_DRG_DATE_LIST")) else []

                    data["predrg_min"]      = "|".join(pre_drg_list)
                    data["predrg_date_min"] = "|".join(pre_date_list)

                    # initialize chosen predrg to empty to allow moving files when none selected
                    data["predrg_max"]      = ""
                    data["predrg_date_max"] = ""

            # ---- CSV meta (dates): copy raw + ISO into JSON if available
            meta_row = lookup_meta_row(meta_df, eds_id, doc_id)
            if meta_row is not None:
                upd_beg_iso = meta_row.get("update_begin_date_iso", "")
                upd_end_iso = meta_row.get("update_end_date_iso", "")
                upd_beg_raw = meta_row.get("update_begin_date_raw", "")
                upd_end_raw = meta_row.get("update_end_date_raw", "")

                if upd_beg_raw:
                    data["update_begin_date_raw"] = str(upd_beg_raw)
                if upd_end_raw:
                    data["update_end_date_raw"] = str(upd_end_raw)

                data["update_begin_date_iso"] = upd_beg_iso
                data["update_end_date_iso"]   = upd_end_iso

                row_data["update_begin_date_iso"] = upd_beg_iso
                row_data["update_end_date_iso"]   = upd_end_iso

            # ---- Choose the pre-DRG closest to update_end_date_iso
            closest_drg, closest_date_iso = closest_drg_to_date(
                pre_drg_list,
                pre_date_list,
                data.get("update_end_date_iso", "")
            )
            if closest_drg:
                data["predrg_max"] = closest_drg
            if closest_date_iso:
                row_data["predrg_max_date"] = closest_date_iso

            # ---- Fill age_years if missing using Oracle AGE_ANNEE
            if data.get("age_years", None) in (None, "", "null") and age_from_oracle is not None:
                data["age_years"] = int(age_from_oracle)

            # update log row summary fields
            row_data.update({
                'drg_target': data.get("drg_target",""),
                'predrg_max': data.get("predrg_max",""),
                'predrg_max_date': row_data.get("predrg_max_date",""),
                'age_years': data.get("age_years", "")
            })

            # ---------- CLEAN-UP: remove technical keys we don't want to persist ----------
            # Keep this list in sync with callers / downstream expectations.
            for k in ("predrg_min_list", "predrg_date_min_list",
                      "update_begin_date", "update_end_date",
                      "predrg_max_date"):
                if k in data:
                    del data[k]
            # ---------------------------------------------------------------------------

            # write formatted JSON to output directory
            dest_path = os.path.join(output_directory, filename)
            with open(dest_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            if not data.get("predrg_max") and stub_mode and data.get("drg_target"):
                data["predrg_max"] = str(data.get("drg_target"))
                row_data["predrg_max"] = data["predrg_max"]
                row_data["status"] = "OK_STUB"

            # ---- Move to no_drg_dir when no chosen pre-DRG exists
            # This marks items needing manual review or further processing.
            if not data.get("predrg_max"):
                target_path = os.path.join(no_drg_dir, filename)
                try:
                    os.makedirs(no_drg_dir, exist_ok=True)

                    # avoid Windows "target exists" error
                    if os.path.exists(target_path):
                        os.remove(target_path)

                    source_moved = None
                    if os.path.exists(dest_path):
                        # move the formatted version (preferred)
                        shutil.move(dest_path, target_path)
                        source_moved = dest_path
                    elif os.path.exists(path):
                        # fallback: move the original JSON
                        shutil.move(path, target_path)
                        source_moved = path
                    else:
                        raise FileNotFoundError(f"Aucune source à déplacer: {dest_path} ni {path}")

                    # remove leftover copy if present
                    leftover = path if source_moved == dest_path else dest_path
                    if os.path.exists(leftover):
                        try:
                            os.remove(leftover)
                        except Exception:
                            # best-effort cleanup — ignore failures
                            pass

                    row_data["status"] = "NO_DRG_MOVED"
                    row_data["message"] = f"Déplacé vers {os.path.relpath(no_drg_dir, output_directory)}"
                except Exception as e:
                    row_data["status"] = "NO_DRG_MOVE_ERROR"
                    row_data["message"] = f"Echec déplacement: {e}"
            else:
                row_data["status"] = "OK"

        except Exception as e:
            # record error and continue processing other files (robust batch behavior)
            row_data['status'] = 'Erreur'
            row_data['message'] = str(e)

        log_rows.append(row_data)

    # write summary CSV log (overwrite)
    if log_rows:
        with open(log_csv_path, "w", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)

# =========================
# Main (standalone convenience)
# =========================
if __name__ == "__main__":
    # --- I/O (example local paths) ---
    input_directory  = "D:/Projets/DRG-Prediction/data/Extractions/RAW-JSON-TEST"
    output_directory = "D:/Projets/DRG-Prediction/data/Extractions/FORMATED-JSON-TEST"
    #input_directory  = "D:/Projets/DRG-Prediction/data/Extractions/RAW-JSON-2025"
    #output_directory = "D:/Projets/DRG-Prediction/data/Extractions/FORMATED-JSON-2025"

    # --- Oracle connection (example / replace with secrets management in production) ---
    oracle_dsn = "oraprdwuni-1.hcuge.ch:1521/pr_dwuni_rw.hcuge.ch"
    oracle_user = "DWHUGOINT_U"
    oracle_password = "mds548w"

    # --- CSV meta (example) ---
    meta_csv_path = "D:/Projets/DRG-Prediction/data/Extractions/DPIDATA-3349/DPIDATA-3349-document_admission.csv"

    # --- GO ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_directory, exist_ok=True)
    log_path = os.path.join(output_directory, f"log_update_{ts}.csv")
    sql_path = DEBUG_SQL_PATH or os.path.join(output_directory, f"oracle_query_{ts}.sql")

    # JSON sources
    eds_ids, json_files = collect_eds_ids(input_directory)

    # Oracle (+ optional SQL debug)
    df_oracle = fetch_oracle_data(
        eds_ids,
        oracle_dsn,
        oracle_user,
        oracle_password,
        batch_size=5000,
        debug_print_sql=DEBUG_PRINT_SQL,
        debug_sql_path=sql_path,
        instant_client_dir=DEFAULT_INSTANT_CLIENT_DIR,
    )

    # CSV meta (robust loader)
    meta_df = load_metadata_csv(meta_csv_path)

    # Update JSONs + log + move cases without a selected pre-DRG
    update_json_files(json_files, df_oracle, meta_df, output_directory, log_path)

    print(f"✅ Terminé. Log : {log_path}")

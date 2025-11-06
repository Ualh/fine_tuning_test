#!/usr/bin/env python3
Preprocessing script for DRG classification datasets.
Overview
--------
This module implements a full preprocessing pipeline for a collection of JSON
medical notes to prepare a classification dataset mapping clinical notes to
DRG labels. It is intended to be used as a CLI tool that:
- Recursively loads JSON files from a source directory.
- Extracts text and DRG labels (with optional truncation).
- Filters examples by token count and optional DRG prefix lists.
- Optionally maps low-frequency classes to a special "__OTHER__" label.
- Splits data into stratified train/validation sets (when possible).
- Computes label-to-id mapping from the training set and writes outputs
    (parquet splits, JSON metadata/statistics, optional copied files).
Data assumptions
----------------
Each input JSON file should contain at least:
- A textual field named "text" or "note" (any other field will be converted
    to string if present).
- A DRG label under one of the candidate keys: "predrg_max" or "drg_target".
- Optionally an identifier field "eds_id" or "stay_id". If missing, the filename
    (without extension) is used as the id.
Key behaviors
-------------
- DRG label selection: choose_drg inspects candidate keys in order and returns
    the first non-empty string. If --truncate_label_to is set, truncation happens
    immediately (before any prefix-based filtering).
- Prefix filtering: when --keep_prefixes is provided, only labels matching any
    of the prefixes are kept, unless --outside_to_other is true in which case
    non-matching labels are mapped to "__OTHER__".
- Token counting: tokenize_lengths uses a Hugging Face AutoTokenizer to count
    token lengths without adding special tokens and without truncation (so the
    raw input length in tokens is measured).
- Min tokens: examples with fewer than --min_tokens tokens are removed.
- Train/validation split: attempts stratified split by label; labels that occur
    only once are held aside and appended to the training set (so the stratify
    call can succeed on labels with at least 2 examples).
- Rare label grouping: if --min_count > 0, labels with frequency < min_count
    in the training split are replaced by "__OTHER__" in both train and val.
- label2id is computed from the final training labels (after potential grouping)
    and applied to validation; any validation examples whose label is not present
    in train are removed.
- Outputs include train.parquet, val.parquet, label2id.json, stats.json,
    meta.json, frequent_labels.json (optional) and optionally copies of original
    JSON files for each split.
Functions (brief)
-----------------
- ensure_dir(p: str) -> str
    Create directory p if missing and return the path (convenience wrapper).
- copy_split_files(df: pd.DataFrame, dst_dir: str) -> int
    Copies original JSON files listed in df["path"] to dst_dir. Returns the
    number of successful copies. Non-fatal: ignores copy errors.
- choose_drg(obj: dict, truncate_label_to: Optional[int]) -> Optional[str]
    Extracts and returns the DRG string from an input JSON object. Applies
    truncation (if requested) before any prefix filtering so that filtering can
    operate on the truncated label.
- load_json_dir(dir_path: str, truncate_label_to: Optional[int]) -> pd.DataFrame
    Recursively finds JSON files in dir_path and builds a DataFrame with columns
    ["text", "drg", "stay_id", "path"] for all valid examples. Filters out:
    - files that cannot be parsed as JSON,
    - examples with empty text,
    - examples lacking a usable DRG label.
    Also prints simple counters: total/ok/empty_text/no_drg.
- tokenize_lengths(texts: List[str], tokenizer_name: str, cutoff_len: int, hf_token: Optional[str]) -> List[int]
    Returns token counts for each input text using AutoTokenizer.from_pretrained.
    The tokenizer is initialized with model_max_length=cutoff_len and can accept
    an HF token via the HF_TOKEN environment variable. Tokenization is done with
    truncation=False and add_special_tokens=False to measure raw length.
- main()
    CLI entrypoint. Parses arguments, orchestrates the full pipeline described
    above, and writes outputs to the provided output directory.
CLI highlights
--------------
Important command-line options and their effects:
- --src (required): input directory with JSONs (recursive).
- --out (required): output directory; created if missing.
- --test_size (float): fraction used for validation set (default 0.1).
- --random_state (int): seed for reproducible splits.
- --min_count (int): labels with fewer examples than this in TRAIN are mapped
    to "__OTHER__" (0 disables).
- --min_tokens (int): minimum token length to keep an example.
- --cutoff_len (int): tokenizer model_max_length (used only to initialize the
    tokenizer and for consistent tokenization behavior).
- --tokenizer (str): HF model identifier for tokenizer (default: a Mistral model).
- --save_splits (bool): controls copying original JSONs into train_split/val_split.
- --keep_prefixes (str): comma-separated DRG prefixes to filter/mask.
- --outside_to_other (bool): when true, labels not matching keep_prefixes are
    mapped to "__OTHER__" instead of being dropped.
- --truncate_label_to (int): truncate DRG labels to this many characters BEFORE
    any prefix filtering/mapping.
Outputs
-------
- train.parquet, val.parquet: final splits with columns
    ["text", "drg", "label", "stay_id", "path", "n_tokens"].
- label2id.json: mapping label -> integer id based on training labels.
- frequent_labels.json: (optional) mapping of frequent label -> raw counts in
    training set when --min_count > 0.
- stats.json: summary statistics of the run (counts, parameters, date).
- meta.json: compact metadata describing CLI parameters used for preprocessing.
- train_split/, val_split/: optional copies of original JSON files for each split.
Notes & warnings
----------------
- The module patches requests.Session.request to set verify=False and disables
    urllib3 insecure request warnings. This disables TLS verification globally
    and is a security risk in production. The patch exists to simplify access to
    private HF token endpoints in some environments but should be removed or made
    configurable for secure deployments.
- Token counting uses the HF tokenizer specified by --tokenizer. If the chosen
    tokenizer is large or requires authentication, tokenization may be slow or
    require a valid HF_TOKEN environment variable.
- The script intentionally measures token counts without adding special tokens
    (add_special_tokens=False) so recorded n_tokens corresponds to raw tokenized
    content length.
- Truncation via --truncate_label_to is applied before filtering and grouping,
    so mapping decisions (e.g., keep_prefixes) operate on the truncated label.
- Rare labels handling: labels that appear only once across the whole dataset
    are not used for stratification; they are appended to the training split to
    preserve them rather than cause stratify to fail.
Example (conceptual)
--------------------
python preprocess_drg.py --src /path/to/jsons --out /tmp/prep_out \
        --tokenizer mistralai/Mistral-7B-v0.1 --min_tokens 40 --test_size 0.1 \
        --keep_prefixes F62,E77 --outside_to_other --min_count 5 --truncate_label_to 3
This will:
- truncate labels to first 3 characters,
- map non-F62/E77 truncated labels to "__OTHER__",
- group labels in train with fewer than 5 occurrences into "__OTHER__",
- create a stratified train/val split when possible, and save outputs to /tmp/prep_out.
# -*- coding: utf-8 -*-
import os, json, glob, argparse, shutil, re
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import numpy as np

# ================ PROXY / SSL ================
import urllib3, requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_original_request = requests.Session.request
def _patched_request(self, *args, **kwargs):
    kwargs["verify"] = False
    return _original_request(self, *args, **kwargs)
requests.Session.request = _patched_request
# ============================================

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

CAND_LABEL_KEYS = ["predrg_max", "drg_target"]


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def copy_split_files(df: pd.DataFrame, dst_dir: str) -> int:
    os.makedirs(dst_dir, exist_ok=True)
    n = 0
    for p in df["path"]:
        try:
            shutil.copy2(p, os.path.join(dst_dir, os.path.basename(p)))
            n += 1
        except Exception:
            pass
    return n


def choose_drg(obj: dict, truncate_label_to: Optional[int] = None) -> Optional[str]:
    """
    Retourne le DRG (priorité: predrg_max, drg_target).
    Si truncate_label_to est défini, tronque AVANT tout filtrage (ex: 3 -> 'F62A' -> 'F62').
    """
    for k in CAND_LABEL_KEYS:
        v = obj.get(k)
        if isinstance(v, str):
            cleaned = v.strip()
            if not cleaned:
                continue
            if truncate_label_to is not None and truncate_label_to > 0 and len(cleaned) >= truncate_label_to:
                return cleaned[:truncate_label_to]
            return cleaned
    return None


def load_json_dir(dir_path: str, truncate_label_to: Optional[int]) -> pd.DataFrame:
    rows = []
    files = sorted(glob.glob(os.path.join(dir_path, "**", "*.json"), recursive=True))
    n_total, n_empty, n_no_drg, n_ok = 0, 0, 0, 0
    for fp in files:
        n_total += 1
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue
        text = obj.get("text") or obj.get("note") or ""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        if not text.strip():
            n_empty += 1
            continue

        drg = choose_drg(obj, truncate_label_to=truncate_label_to)
        if drg is None or not isinstance(drg, str) or not drg.strip():
            n_no_drg += 1
            continue

        stay_id = obj.get("eds_id") or obj.get("stay_id") or os.path.splitext(os.path.basename(fp))[0]
        rows.append({"text": text, "drg": drg.strip(), "stay_id": str(stay_id), "path": fp})
        n_ok += 1
    print(f"[load_json_dir] dir={dir_path} | total={n_total} | ok={n_ok} | no_drg={n_no_drg} | empty_text={n_empty}")
    return pd.DataFrame(rows)


def tokenize_lengths(texts: List[str], tokenizer_name: str, cutoff_len: int, hf_token: Optional[str]) -> List[int]:
    tok = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=cutoff_len, token=hf_token)
    lengths = []
    for t in texts:
        ids = tok(t, truncation=False, add_special_tokens=False)["input_ids"]
        lengths.append(len(ids))
    return lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True, help="Répertoire des JSON bruts")
    ap.add_argument("--out", type=str, required=True, help="Dossier de sortie du prétraitement (sera créé)")
    ap.add_argument("--test_size", type=float, default=0.1, help="Taille du split validation (ex. 0.1=10%)")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--min_count", type=int, default=0, help="Regrouper <min_count en __OTHER__ (0 pour désactiver)")
    ap.add_argument("--min_tokens", type=int, default=40, help="Exclure les textes avec moins de tokens")
    ap.add_argument("--cutoff_len", type=int, default=2048, help="Longueur max visée (pour compter les tokens)")
    ap.add_argument("--tokenizer", type=str, default="mistralai/Mistral-7B-v0.1", help="Tokenizer pour compter les tokens")
    ap.add_argument("--save_splits", type=lambda x: str(x).lower() in {"1","true","yes","y"}, default=True)

    # Filtrage par préfixes
    ap.add_argument(
        "--keep_prefixes",
        type=str,
        default="",
        help="Liste de préfixes DRG à conserver, séparés par des virgules (ex: 'F62,E77,E86'). "
             "Les classes hors liste seront supprimées OU mappées en __OTHER__ si --outside_to_other true."
    )
    ap.add_argument(
        "--outside_to_other",
        type=lambda x: str(x).lower() in {"1","true","yes","y"},
        default=False,
        help="Si true, les classes hors préfixes sont mappées à __OTHER__ (au lieu d'être supprimées)."
    )

    # *** NOUVEAU : tronquer les DRG avant filtrage ***
    ap.add_argument(
        "--truncate_label_to",
        type=int,
        default=None,
        help="Si défini, tronque les DRG à ce nombre de caractères (ex: 3 => 'F62A' -> 'F62')."
    )

    args = ap.parse_args()
    HF_TOKEN = os.environ.get("HF_TOKEN")
    out_dir = ensure_dir(args.out)

    # 1) Charger tous les JSON (avec tronquage anticipé)
    df_all = load_json_dir(args.src, truncate_label_to=args.truncate_label_to)
    if df_all.empty:
        raise ValueError(f"Aucun JSON exploitable dans {args.src}")

    # 1bis) Filtrage par préfixes (sur labels déjà tronqués)
    keep_list = [p.strip() for p in args.keep_prefixes.split(",") if p.strip()]
    if keep_list:
        prefix_pat = re.compile(r"^(%s)" % "|".join(map(re.escape, keep_list)))
        before = len(df_all)
        if args.outside_to_other:
            mask_keep = df_all["drg"].astype(str).str.match(prefix_pat)
            df_all.loc[~mask_keep, "drg"] = "__OTHER__"
            print(f"[keep_prefixes] outside_to_other=True | prefixes={keep_list} | "
                  f"mapped_outside_to='__OTHER__' | total={len(df_all)}")
        else:
            df_all = df_all[df_all["drg"].astype(str).str.match(prefix_pat)].copy()
            print(f"[keep_prefixes] prefixes={keep_list} | kept={len(df_all)} (dropped={before-len(df_all)})")
        if df_all.empty:
            raise ValueError("[keep_prefixes] Après filtrage/mapping, plus aucune donnée exploitable.")

    # 2) Compter les tokens et filtrer les trop courts
    print("[tokens] comptage des longueurs…")
    df_all["n_tokens"] = tokenize_lengths(df_all["text"].tolist(), args.tokenizer, args.cutoff_len, HF_TOKEN)
    before_len = len(df_all)
    df_all = df_all[df_all["n_tokens"] >= args.min_tokens].copy()
    print(f"[tokens] filtrés (min_tokens={args.min_tokens}) : {before_len - len(df_all)} supprimés")

    # 3) Split train/val (stratifié si possible)
    counts = df_all["drg"].value_counts()
    rare_for_strata = counts[counts < 2].index.tolist()
    df_common = df_all[~df_all["drg"].isin(rare_for_strata)].copy()
    df_rare = df_all[df_all["drg"].isin(rare_for_strata)].copy()
    if len(df_common) == 0:
        tr_idx, va_idx = train_test_split(
            np.arange(len(df_all)), test_size=args.test_size, random_state=args.random_state, shuffle=True
        )
        df_train = df_all.iloc[tr_idx].copy()
        df_val = df_all.iloc[va_idx].copy()
    else:
        train_c, val_c = train_test_split(
            df_common, test_size=args.test_size, random_state=args.random_state, stratify=df_common["drg"]
        )
        df_train = pd.concat([train_c, df_rare], ignore_index=True)
        df_val = val_c.copy()

    # 4) Regrouper les classes rares (basé sur le TRAIN uniquement)
    frequent_json = {}
    if args.min_count > 0:
        counts_train = df_train["drg"].value_counts()
        frequent = set(counts_train[counts_train >= args.min_count].index.tolist())
        def map_other(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df.loc[~df["drg"].isin(frequent), "drg"] = "__OTHER__"
            return df
        df_train = map_other(df_train)
        df_val = map_other(df_val)
        frequent_json = {lbl: int(counts_train.get(lbl, 0)) for lbl in sorted(list(frequent))}
        print(f"[OTHER] min_count={args.min_count} | frequent={len(frequent)} | '__OTHER__' activé")

    # 5) Construire label2id depuis TRAIN et mapper
    drg_unique = sorted(df_train["drg"].unique().tolist())
    label2id = {d: i for i, d in enumerate(drg_unique)}
    df_train["label"] = df_train["drg"].map(label2id).astype(int)
    df_val = df_val[df_val["drg"].isin(label2id.keys())].copy()
    df_val["label"] = df_val["drg"].map(label2id).astype(int)

    # 6) Sauvegardes
    df_train_out = df_train[["text", "drg", "label", "stay_id", "path", "n_tokens"]].reset_index(drop=True)
    df_val_out = df_val[["text", "drg", "label", "stay_id", "path", "n_tokens"]].reset_index(drop=True)

    df_train_out.to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
    df_val_out.to_parquet(os.path.join(out_dir, "val.parquet"), index=False)
    with open(os.path.join(out_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    if frequent_json:
        with open(os.path.join(out_dir, "frequent_labels.json"), "w", encoding="utf-8") as f:
            json.dump(frequent_json, f, ensure_ascii=False, indent=2)

    stats = {
        "src": args.src,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "min_count": args.min_count,
        "min_tokens": args.min_tokens,
        "cutoff_len": args.cutoff_len,
        "tokenizer": args.tokenizer,
        "n_all": int(len(df_all)),
        "n_train": int(len(df_train_out)),
        "n_val": int(len(df_val_out)),
        "n_classes": int(len(label2id)),
        "has_OTHER": "__OTHER__" in label2id,
        "keep_prefixes": keep_list,
        "outside_to_other": bool(args.outside_to_other),
        "truncate_label_to": args.truncate_label_to,
        "date": datetime.now().isoformat(timespec="seconds")
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    if args.save_splits:
        tr_dir = ensure_dir(os.path.join(out_dir, "train_split"))
        va_dir = ensure_dir(os.path.join(out_dir, "val_split"))
        ntr = copy_split_files(df_train_out, tr_dir)
        nva = copy_split_files(df_val_out, va_dir)
        print(f"[split] train_copied={ntr} → {tr_dir} | val_copied={nva} → {va_dir}")

    meta = {
        "seed": args.random_state,
        "test_size": args.test_size,
        "min_count": args.min_count,
        "min_tokens": args.min_tokens,
        "tokenizer": args.tokenizer,
        "cutoff_len": args.cutoff_len,
        "save_splits": bool(args.save_splits),
        "keep_prefixes": keep_list,
        "outside_to_other": bool(args.outside_to_other),
        "truncate_label_to": args.truncate_label_to,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Prétraitement → {out_dir}")


if __name__ == "__main__":
    main()


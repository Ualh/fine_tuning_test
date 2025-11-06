#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, glob, argparse, re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # rendu headless
import matplotlib.pyplot as plt

# Proxy/SSL tolérant (optionnel, comme ton setup)
import urllib3, requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_original_request = requests.Session.request
def _patched_request(self, *args, **kwargs):
    kwargs["verify"] = False
    return _original_request(self, *args, **kwargs)
requests.Session.request = _patched_request

from transformers import AutoTokenizer

CAND_LABEL_KEYS = [
    "drg_target", "predrg_min", "drg", "label", "predrg", "primary_drg"
]

def pick_label(obj: dict) -> str | None:
    for k in CAND_LABEL_KEYS:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def iter_json_files(root: str):
    pattern = os.path.join(root, "**", "*.json")
    for fp in glob.iglob(pattern, recursive=True):
        yield fp

def percentile_summary(arr: np.ndarray, name: str) -> str:
    if arr.size == 0:
        return f"[{name}] aucun échantillon"
    qs = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])
    return (f"[{name}] n={arr.size} | "
            f"min={int(arr.min())} | p1={int(qs[0])} | p5={int(qs[1])} | "
            f"p25={int(qs[2])} | p50={int(qs[3])} | p75={int(qs[4])} | "
            f"p95={int(qs[5])} | p99={int(qs[6])} | max={int(arr.max())}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Répertoire racine contenant les JSON (récursif).")
    ap.add_argument("--out_dir", default="./outputs/stats", help="Dossier de sortie (CSV/PNG).")
    ap.add_argument("--tokenizer", default="mistralai/Mistral-7B-v0.1", help="Tokenizer HF pour le comptage de tokens.")
    ap.add_argument("--cutoff_len", type=int, default=2048, help="model_max_length du tokenizer.")
    ap.add_argument("--cache_dir", default=None, help="Cache HF optionnel.")
    ap.add_argument("--max_files", type=int, default=0, help="Limiter le nombre de JSON lus (0 = tous).")
    ap.add_argument("--bins_chars", type=int, default=80, help="Nb de bacs pour l’histogramme caractères.")
    ap.add_argument("--bins_tokens", type=int, default=80, help="Nb de bacs pour l’histogramme tokens.")
    ap.add_argument("--title_suffix", default="", help="Suffixe pour les titres de figures.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    HF_TOKEN = os.environ.get("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(
        args.tokenizer, model_max_length=args.cutoff_len, cache_dir=args.cache_dir, token=HF_TOKEN
    )

    rows = []
    misses_text = 0
    misses_label = 0
    total = 0

    for i, fp in enumerate(iter_json_files(args.data_dir)):
        if args.max_files and i >= args.max_files:
            break
        total += 1
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        text = obj.get("text") or obj.get("note") or ""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        label = pick_label(obj)
        if not text.strip():
            misses_text += 1
        if label is None:
            misses_label += 1

        # longueurs
        char_len = len(text)
        try:
            ids = tok(text, truncation=False, add_special_tokens=False)["input_ids"]
            tok_len = len(ids)
        except Exception:
            tok_len = 0

        rows.append({
            "path": fp,
            "drg": label if label is not None else "",
            "char_len": char_len,
            "tok_len": tok_len
        })

    df = pd.DataFrame(rows)
    print(f"[PARSE] fichiers lus={total} | lignes retenues={len(df)} | sans texte={misses_text} | sans DRG={misses_label}")

    # Comptage DRG
    drg_counts = df["drg"].value_counts(dropna=False)
    counts_csv = os.path.join(args.out_dir, "drg_counts.csv")
    drg_counts.to_csv(counts_csv, header=["count"])
    print(f"[OUT] Comptages DRG → {counts_csv}")

    # Détail complet
    details_csv = os.path.join(args.out_dir, "notes_lengths.csv")
    df.to_csv(details_csv, index=False)
    print(f"[OUT] Détails longueurs → {details_csv}")

    # Stats descriptives
    char_arr = df["char_len"].to_numpy(dtype=np.int64)
    tok_arr  = df["tok_len"].to_numpy(dtype=np.int64)
    print(percentile_summary(char_arr, "chars"))
    print(percentile_summary(tok_arr,  "tokens"))

    # Histogrammes
    # 1) caractères
    plt.figure(figsize=(10, 5))
    plt.hist(char_arr, bins=args.bins_chars)
    ttl = "Histogramme tailles (caractères)"
    if args.title_suffix:
        ttl += f" — {args.title_suffix}"
    plt.title(ttl)
    plt.xlabel("Nombre de caractères")
    plt.ylabel("Fréquence")
    plt.tight_layout()
    out_png_chars = os.path.join(args.out_dir, "hist_chars.png")
    plt.savefig(out_png_chars, dpi=140)
    plt.close()
    print(f"[OUT] Histogramme caractères → {out_png_chars}")

    # 2) tokens
    plt.figure(figsize=(10, 5))
    plt.hist(tok_arr, bins=args.bins_tokens)
    ttl = "Histogramme tailles (tokens)"
    if args.title_suffix:
        ttl += f" — {args.title_suffix}"
    plt.title(ttl)
    plt.xlabel("Nombre de tokens")
    plt.ylabel("Fréquence")
    plt.tight_layout()
    out_png_tokens = os.path.join(args.out_dir, "hist_tokens.png")
    plt.savefig(out_png_tokens, dpi=140)
    plt.close()
    print(f"[OUT] Histogramme tokens → {out_png_tokens}")

    # Petit top DRG
    top_csv = os.path.join(args.out_dir, "drg_top20.csv")
    drg_counts.head(20).to_csv(top_csv, header=["count"])
    print(f"[OUT] Top 20 DRG → {top_csv}")

if __name__ == "__main__":
    main()

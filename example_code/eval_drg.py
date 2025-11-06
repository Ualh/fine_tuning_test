"""
Evaluation utilities for DRG (Diagnosis-Related Group) classification models.

Stage: evaluation — loads JSON notes, prepares chunked evaluation dataset, runs a
Hugging Face/PEFT classifier (supports 4-bit quantized bases + LoRA adapters),
aggregates chunk-level predictions to group-level (per-stay), computes metrics,
and writes CSV / JSON / confusion outputs.

See: run this module as a script to evaluate a trained run folder that contains
label2id.json and optionally a Peft adapter (model_dir).
"""
import os, json, glob, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, top_k_accuracy_score

# SSL/proxy (enterprise) — disable SSL verification for environments that require
# it (keeps calls to HF or other hosts working behind a corporate proxy).
# This is a best-effort patch used in some CI/enterprise environments; avoid in
# general-purpose production code unless you really need it.
import urllib3, requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_original_request = requests.Session.request
def _patched_request(self, *args, **kwargs):
    kwargs["verify"] = False
    return _original_request(self, *args, **kwargs)
requests.Session.request = _patched_request

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, Trainer, TrainingArguments, BitsAndBytesConfig
)
from peft import PeftModel


def load_json_dir(dir_path: str) -> pd.DataFrame:
    """
    Load JSON files recursively from dir_path and return a DataFrame with
    columns: text, drg, stay_id.

    Behavior:
    - Searches for all .json files under dir_path (recursive).
    - Extracts text from common fields ("text" or "note").
    - Extracts label from common fields ("predrg_max", "drg_target", "label").
    - Uses filename as stay_id fallback.
    - Skips files that cannot be parsed or where label is missing/empty.

    Args:
        dir_path: root directory containing JSON files.

    Returns:
        pd.DataFrame with rows {"text", "drg", "stay_id"}.
    """
    rows = []
    files = sorted(glob.glob(os.path.join(dir_path, "**", "*.json"), recursive=True))
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            # skip unreadable/corrupted JSON file quietly
            continue
        text = obj.get("text") or obj.get("note") or ""
        drg = obj.get("predrg_max") or obj.get("drg_target") or obj.get("label") or None
        stay_id = obj.get("stay_id") or os.path.splitext(os.path.basename(fp))[0]
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        # Skip if no valid DRG label present
        if not isinstance(drg, str) or drg.strip() == "":
            continue
        rows.append({"text": text, "drg": drg.strip(), "stay_id": str(stay_id)})
    return pd.DataFrame(rows)


def build_eval_df(
    data_dir: str,
    label2id_path: str,
    map_unknown_to_other: bool = False
) -> tuple[pd.DataFrame, dict]:
    """
    Build an evaluation DataFrame and load label2id mapping.

    Behavior:
    - Loads JSONs from data_dir via load_json_dir.
    - Loads label2id.json (mapping DRG string -> integer id).
    - Optionally maps unknown DRG labels to the special "__OTHER__" class
      if that class exists in label2id; otherwise filters unknown DRGs out.

    Args:
        data_dir: directory containing JSON notes to evaluate.
        label2id_path: path to label2id.json produced by training.
        map_unknown_to_other: if True and "__OTHER__" exists, map unknown labels
                              to "__OTHER__"; otherwise drop unknown labels.

    Returns:
        Tuple (df, label2id) where df contains columns ["text", "drg", "stay_id", "label"].

    Raises:
        ValueError: if no valid JSON rows were found in data_dir.
    """
    df = load_json_dir(data_dir)
    if df.empty:
        raise ValueError(f"Aucun JSON valide dans {data_dir}")

    with open(label2id_path, "r", encoding="utf-8") as f:
        label2id = json.load(f)

    known = set(label2id.keys())
    has_other = "__OTHER__" in known

    if map_unknown_to_other and has_other:
        # Map unknown DRG values to the special __OTHER__ token
        df["drg"] = df["drg"].apply(lambda d: d if d in known else "__OTHER__")
        df["label"] = df["drg"].map(label2id).astype(int)
    else:
        # Keep only rows whose DRG is known to the model
        df = df[df["drg"].isin(known)].copy()
        df["label"] = df["drg"].map(label2id).astype(int)

    return df, label2id


def make_chunks(ds: Dataset, tokenizer: AutoTokenizer, cutoff_len: int, min_tokens: int):
    """
    Convert a Dataset of full texts into a tokenized chunked Dataset suitable
    for model evaluation.

    Steps:
    1) Filter out very short notes (less than min_tokens tokens).
    2) Split long notes into overlapping chunks (sliding window) with CHUNK and STRIDE.
       This produces multiple (text, label, group_id) rows per original note.
    3) Tokenize each chunk with truncation to cutoff_len and rename "label" -> "labels"
       to match HF Trainer expectation.

    Args:
        ds: HuggingFace Dataset with columns at least ["text", "label", "stay_id"].
        tokenizer: HF tokenizer used for token counting/decoding.
        cutoff_len: model maximum token length (final truncation length).
        min_tokens: minimum number of tokens to keep a note for chunking.

    Returns:
        Tokenized HF Dataset with columns including "input_ids", "attention_mask", "labels", and "group_id".
    """
    # 1) Filter short notes based on raw token count (no special tokens)
    def keep(batch):
        toks = tokenizer(batch["text"], truncation=False, add_special_tokens=False)
        return {"keep": [len(x) >= min_tokens for x in toks["input_ids"]]}
    ds = ds.map(keep, batched=True).filter(lambda x: x["keep"])

    # 2) Sliding-window chunking: CHUNK is the effective chunk size, STRIDE controls overlap.
    def expand(batch):
        out_text, out_label, out_gid = [], [], []
        CHUNK = max(128, cutoff_len - 16)  # keep some margin from cutoff to allow special tokens
        STRIDE = max(64, CHUNK // 4)
        gids = batch["stay_id"] if "stay_id" in batch else batch["__index_level_0__"]
        for t, y, gid in zip(batch["text"], batch["label"], gids):
            if not t or not str(t).strip():
                continue
            ids = tokenizer(t, truncation=False, add_special_tokens=False)["input_ids"]
            start = 0
            while start < len(ids):
                end = min(start + CHUNK, len(ids))
                chunk_ids = ids[start:end]
                # decode without adding/removing special tokens so text aligns with tokenization step later
                out_text.append(tokenizer.decode(chunk_ids, skip_special_tokens=False))
                out_label.append(int(y))
                out_gid.append(str(gid))
                if end == len(ids):
                    break
                # overlap by STRIDE to create context continuity between chunks
                start = end - STRIDE
        return {"text": out_text, "label": out_label, "group_id": out_gid}
    ds = ds.map(expand, batched=True, remove_columns=ds.column_names)

    # 3) Final tokenization with explicit truncation to cutoff_len
    def preprocess(batch):
        return tokenizer(batch["text"], truncation=True, max_length=cutoff_len)
    ds_tok = ds.map(preprocess, batched=True).remove_columns(["text"]).rename_column("label", "labels")
    return ds_tok


def compute_metrics_np(pred, labels, C):
    """
    Compute basic classification metrics from raw predictions and labels.

    Args:
        pred: numpy array shape [N, C] of logits or probabilities.
        labels: 1-D ground-truth label array length N.
        C: number of classes.

    Returns:
        dict with keys 'acc', 'balanced_acc', 'macro_f1', and optional 'top3'/'top5'.

    Notes:
        Uses argmax to compute hard predictions for accuracy/F1. top-k metrics
        are computed from raw scores and may fail for degenerate inputs (caught).
    """
    out = {
        "acc": float(accuracy_score(labels, pred.argmax(axis=1))),
        "balanced_acc": float(balanced_accuracy_score(labels, pred.argmax(axis=1))),
        "macro_f1": float(f1_score(labels, pred.argmax(axis=1), average="macro")),
    }
    try:
        out["top3"] = float(top_k_accuracy_score(labels, pred, k=3))
        out["top5"] = float(top_k_accuracy_score(labels, pred, k=5)) if C >= 5 else None
    except Exception:
        # best-effort: ignore top-k failures rather than crashing evaluation
        pass
    return out


def main():
    """
    Command-line entrypoint for evaluation.

    Typical usage:
      python eval_drg.py --data_dir prepared/val --model_dir outputs/run-xxx --base_model mistralai/Mistral-7B-v0.1

    Key behaviors:
    - Loads label2id.json from model_dir to determine class space.
    - Loads and chunk-ifies JSON notes from data_dir.
    - Loads base model + PEFT adapter (supports BitsAndBytes 4-bit config).
    - Runs HF Trainer.predict to obtain logits per chunk.
    - Aggregates chunk logits to group-level (per stay_id) by mean pooling.
    - Optionally performs abstention by thresholding max probability and mapping to __OTHER__.
    - Writes grouped predictions CSV, metrics JSON, and confusion matrices/heatmap.

    CLI Args:
        See argparse configuration in the function.

    Returns:
        None (writes outputs to model_dir).
    """
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="Dossier JSON à évaluer (val_split ou équivalent)")
    p.add_argument("--model_dir", type=str, required=True, help="Dossier du run fine-tuning (contient label2id.json)")
    p.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--cutoff_len", type=int, default=1024)
    p.add_argument("--min_tokens", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--cache_dir", type=str, default=None)
    # --- nouveaux paramètres ---
    p.add_argument("--abstain_tau", type=float, default=None,
                   help="Seuil d'abstention (0..1) appliqué après agrégation; si None, pas d'abstention")
    p.add_argument("--map_unknown_to_other", type=str, default="false",
                   help="true/false : mapper les DRG inconnus vers __OTHER__ si disponible")
    p.add_argument("--topk_heatmap", type=int, default=30, help="K pour la heatmap de confusion top-K classes")
    args = p.parse_args()

    map_unknown = str(args.map_unknown_to_other).lower() in {"1", "true", "yes", "y"}

    label2id_path = os.path.join(args.model_dir, "label2id.json")
    df_eval, label2id = build_eval_df(args.data_dir, label2id_path, map_unknown_to_other=map_unknown)
    num_labels = len(label2id)
    id2label = {v: k for k, v in label2id.items()}
    print(f"[EVAL] {len(df_eval)} notes | {num_labels} classes | abstain_tau={args.abstain_tau} | map_unknown={map_unknown}")

    # Model loading: support 4-bit quantized base model + PEFT adapter
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
    )
    HF_TOKEN = os.environ.get("HF_TOKEN")
    # from_pretrained may download model files; cache_dir and token are forwarded
    model_base = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=num_labels, id2label=id2label, label2id=label2id,
        quantization_config=quant, device_map="auto", cache_dir=args.cache_dir, token=HF_TOKEN
    )
    model = PeftModel.from_pretrained(model_base, args.model_dir, is_trainable=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, model_max_length=args.cutoff_len, cache_dir=args.cache_dir, token=HF_TOKEN
    )
    if tokenizer.pad_token is None:
        # Some generative tokenizers have no pad_token; set to eos to allow padding
        tokenizer.pad_token = tokenizer.eos_token

    ds_eval = Dataset.from_pandas(df_eval.reset_index(drop=True))
    ds_eval_tok = make_chunks(ds_eval, tokenizer, args.cutoff_len, args.min_tokens)
    val_group_ids = ds_eval_tok["group_id"]

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    args_tr = TrainingArguments(
        output_dir=os.path.join(args.model_dir, "eval_tmp"),
        per_device_eval_batch_size=args.batch_size,
        fp16=True,
        dataloader_drop_last=False,
        report_to=[],
    )
    trainer = Trainer(model=model, args=args_tr, tokenizer=tokenizer, data_collator=data_collator)
    preds = trainer.predict(ds_eval_tok)
    logits = preds.predictions  # [N_chunks, C]
    labels = ds_eval_tok["labels"]

    # --- chunk-level metrics (no abstention) ---
    m_chunk = compute_metrics_np(logits, labels, num_labels)
    print("[Chunk] ", m_chunk)

    # --- aggregate chunk logits to group-level (per stay/note) by averaging ---
    gid_idx = pd.factorize(val_group_ids)[0]  # integer group index per chunk
    logits_t = torch.tensor(logits)
    gid_t = torch.tensor(gid_idx)
    uniq = torch.unique(gid_t)
    agg_logits_t = []
    for g in uniq.tolist():
        # mean pooling across chunks that belong to the same group
        agg_logits_t.append(logits_t[gid_t == g].mean(dim=0, keepdim=True))
    agg_logits_t = torch.cat(agg_logits_t, dim=0)        # [G, C]
    agg_logits = agg_logits_t.cpu().numpy()

    # Ground truth per group: take the label of the first chunk in the group
    first_idx = [int(np.where(gid_idx == g)[0][0]) for g in uniq.numpy()]
    y_true = np.array(labels)[first_idx]
    y_pred = agg_logits.argmax(axis=1)

    # --- optional abstention after aggregation ---
    if args.abstain_tau is not None:
        if "__OTHER__" in label2id:
            other_id = label2id["__OTHER__"]
            probs = F.softmax(agg_logits_t, dim=1).cpu().numpy()
            maxp = probs.max(axis=1)
            abstain_mask = (maxp < float(args.abstain_tau))
            # map abstained groups to the __OTHER__ id
            y_pred = np.where(abstain_mask, other_id, y_pred)
            print(f"[Abstention] tau={args.abstain_tau} | abstentions={int(abstain_mask.sum())}/{len(y_pred)} "
                  f"({100.0*abstain_mask.mean():.2f}%)")
        else:
            print("[Abstention] Ignorée: '__OTHER__' non présent dans label2id.json")

    # --- group-level metrics (after possible abstention) ---
    m_group = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    try:
        m_group["top3"] = float(top_k_accuracy_score(y_true, agg_logits, k=3))
        m_group["top5"] = float(top_k_accuracy_score(y_true, agg_logits, k=5)) if num_labels >= 5 else None
    except Exception:
        pass
    print("[Group]", m_group)

    # --- write main exports: grouped predictions CSV and metrics JSON ---
    out_csv = os.path.join(args.model_dir, "eval_grouped_predictions.csv")
    pd.DataFrame({
        "group_index": uniq.numpy(),
        "true": y_true,
        "pred": y_pred,
        "true_drg": [id2label[int(t)] for t in y_true],
        "pred_drg": [id2label[int(p)] for p in y_pred],
    }).to_csv(out_csv, index=False)
    print(f"[EVAL] CSV écrit -> {out_csv}")

    with open(os.path.join(args.model_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"chunk": m_chunk, "group": m_group}, f, indent=2, ensure_ascii=False)
    print(f"[EVAL] Métriques -> {os.path.join(args.model_dir, 'eval_metrics.json')}")

    # ============= Confusion matrices and heatmap =============
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    cm_dir = os.path.join(args.model_dir, "confusion")
    os.makedirs(cm_dir, exist_ok=True)

    # --- 1) Chunk-level confusion (raw + normalized by true class) ---
    y_chunk_true = np.array(labels)
    y_chunk_pred = logits.argmax(axis=1)

    cm_chunk = confusion_matrix(y_chunk_true, y_chunk_pred, labels=list(range(num_labels)))
    cm_chunk_norm = confusion_matrix(y_chunk_true, y_chunk_pred, labels=list(range(num_labels)), normalize="true")

    pd.DataFrame(cm_chunk, index=[id2label[i] for i in range(num_labels)],
                           columns=[id2label[i] for i in range(num_labels)]).to_csv(
        os.path.join(cm_dir, "confusion_chunk_raw.csv"), encoding="utf-8"
    )
    pd.DataFrame(cm_chunk_norm, index=[id2label[i] for i in range(num_labels)],
                               columns=[id2label[i] for i in range(num_labels)]).to_csv(
        os.path.join(cm_dir, "confusion_chunk_normalized.csv"), encoding="utf-8"
    )

    # --- 2) Group-level confusion (after aggregation and abstention) ---
    cm_group = confusion_matrix(y_true, y_pred, labels=list(range(num_labels)))
    cm_group_norm = confusion_matrix(y_true, y_pred, labels=list(range(num_labels)), normalize="true")

    pd.DataFrame(cm_group, index=[id2label[i] for i in range(num_labels)],
                          columns=[id2label[i] for i in range(num_labels)]).to_csv(
        os.path.join(cm_dir, "confusion_group_raw.csv"), encoding="utf-8"
    )
    pd.DataFrame(cm_group_norm, index=[id2label[i] for i in range(num_labels)],
                              columns=[id2label[i] for i in range(num_labels)]).to_csv(
        os.path.join(cm_dir, "confusion_group_normalized.csv"), encoding="utf-8"
    )

    # --- 3) Top-K heatmap (group-level normalized confusion) ---
    K = int(args.topk_heatmap) if args.topk_heatmap and args.topk_heatmap > 0 else 30
    support = pd.Series(y_true).value_counts().sort_values(ascending=False)
    topk_ids = support.index[:K].tolist()
    topk_labels = [id2label[int(i)] for i in topk_ids]

    cm_group_topk = confusion_matrix(y_true, y_pred, labels=topk_ids, normalize="true")
    plt.figure(figsize=(12, 10))
    plt.imshow(cm_group_topk, interpolation="nearest", aspect="auto")
    plt.title(f"Confusion (group-level) Top-{K} classes (normalisee)")
    plt.xlabel("Prédit")
    plt.ylabel("Vrai")
    plt.xticks(ticks=np.arange(len(topk_labels)), labels=topk_labels, rotation=90)
    plt.yticks(ticks=np.arange(len(topk_labels)), labels=topk_labels)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(cm_dir, f"confusion_group_top{K}.png"), dpi=200)
    plt.close()

    print(f"[EVAL] Confusion matrices écrites dans {cm_dir}")


if __name__ == "__main__":
    main()

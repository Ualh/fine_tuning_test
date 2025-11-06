#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, math
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset

# ===================== Réseau / SSL désactivé =====================
import urllib3, requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_original_request = requests.Session.request
def _patched_request(self, *args, **kwargs):
    kwargs["verify"] = False
    return _original_request(self, *args, **kwargs)
requests.Session.request = _patched_request
# =================================================================

# ============================ W&B offline =========================
import wandb
wandb.init(mode="offline")
# =================================================================

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    BitsAndBytesConfig, DataCollatorWithPadding,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from peft import (
    LoraConfig, get_peft_model, prepare_model_for_kbit_training,
    TaskType
)

# ============================== Focal =============================
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if isinstance(alpha, (list, tuple, np.ndarray)):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha if isinstance(alpha, torch.Tensor) else None)
        self.alpha_scalar = float(alpha) if isinstance(alpha, (int, float)) else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            at = self.alpha.to(logits.device)[targets]
            loss = at * loss
        elif self.alpha_scalar is not None:
            loss = self.alpha_scalar * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
# =================================================================

# ============================== Utils =============================
def create_folder(base: str, name: str) -> str:
    path = os.path.join(base, name)
    os.makedirs(path, exist_ok=True)
    return path

def reinit_classifier_head(m) -> bool:
    """Réinitialise le head de classification si présent (classifier/score)."""
    done = False
    for attr in ["classifier", "score"]:
        if hasattr(m, attr):
            head = getattr(m, attr)
            for p in head.parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)
                else:
                    torch.nn.init.zeros_(p)
            print(f"[Init] Reinit head: {attr}")
            done = True
    return done
# =================================================================


def main():
    # Évite le warning "tokenizers got forked..."
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # ==== Diagnostics CUDA ====
    print("[CUDA] is_available =", torch.cuda.is_available())
    try:
        if torch.cuda.is_available():
            print("[CUDA] device_count =", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                print(f"[CUDA] device[{i}] =", torch.cuda.get_device_name(i))
        else:
            print("[CUDA] GPU non disponible — attention aux performances.")
    except Exception as e:
        print("[CUDA] check error:", e)

    # ==================== Arguments ====================
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Dossier de prétraitement (train.parquet, val.parquet, label2id.json)")
    ap.add_argument("--output_base", type=str, default="./outputs")

    # modèle / tokenisation
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2-7B")
    ap.add_argument("--cutoff_len", type=int, default=8192)
    ap.add_argument("--padding_side", type=str, default="right")

    # optim
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--min_lr", type=float, default=5e-6, help="LR plancher pour le cosine scheduler")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lr_scheduler_type", type=str, default="cosine")
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_batch_size", type=int, default=None)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_targets", type=str, default="q_proj,k_proj,v_proj,o_proj")

    # pertes / déséquilibre
    ap.add_argument("--loss_type", type=str, default="weighted_ce", choices=["weighted_ce", "focal"])
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--use_oversampling", action="store_true", help="Active WeightedRandomSampler sur le train")

    # sélection de modèle / early stopping
    ap.add_argument("--metric_for_best_model", type=str, default="macro_f1")
    ap.add_argument("--greater_is_better", type=lambda x: str(x).lower()=="true", default=True)
    ap.add_argument("--early_stopping_patience", type=int, default=2)

    # sanity / head
    ap.add_argument("--reinit_head", action="store_true", help="Réinitialise le head au démarrage")
    ap.add_argument("--freeze_base_for_sanity", action="store_true", help="Gèle la base pour entraîner seulement le head")

    # divers
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--resume_from", type=str, default=None)
    ap.add_argument("--wandb_project", type=str, default="classification")
    ap.add_argument("--wandb_watch", type=str, default="gradients")
    ap.add_argument("--wandb_log_model", type=str, default="")
    ap.add_argument("--text_column", type=str, default=None, help="Nom explicite de la colonne texte si différente de 'text'")

    # abstention (facultatif)
    ap.add_argument("--abstain_tau", type=float, default=None)

    args = ap.parse_args()
    HF_TOKEN = os.environ.get("HF_TOKEN")

    # ================== Chargement data ==================
    train_pq = os.path.join(args.data_dir, "train.parquet")
    val_pq   = os.path.join(args.data_dir, "val.parquet")
    l2i_path = os.path.join(args.data_dir, "label2id.json")
    if not (os.path.isfile(train_pq) and os.path.isfile(val_pq) and os.path.isfile(l2i_path)):
        raise FileNotFoundError("train.parquet / val.parquet / label2id.json introuvables dans --data_dir")

    df_train = pd.read_parquet(train_pq)
    df_val   = pd.read_parquet(val_pq)
    with open(l2i_path, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    num_labels = len(label2id)
    id2label = {int(v): k for k, v in label2id.items()}

    # dossier output du run
    now = datetime.now().strftime("%b-%d-%H-%M")
    run_name = f"7b-{args.cutoff_len}-{args.batch_size}x{args.grad_accum}-{args.loss_type}-{now}"
    output_dir = create_folder(args.output_base, run_name)
    with open(os.path.join(output_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    print(f"[DATA] Train={len(df_train)} Val={len(df_val)} Classes={num_labels}")
    print(f"[OUT ] {output_dir}")

    # DDP & W&B
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    if args.wandb_project: os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_watch:   os.environ["WANDB_WATCH"] = args.wandb_watch
    if args.wandb_log_model: os.environ["WANDB_LOG_MODEL"] = args.wandb_log_model

    # A6000 flags
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # ================== Modèle 4-bit ==================
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        quantization_config=quant,
        device_map=device_map,
        cache_dir=args.cache_dir,
        token=HF_TOKEN
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        model_max_length=args.cutoff_len,
        cache_dir=args.cache_dir,
        token=HF_TOKEN
    )
    tokenizer.padding_side = args.padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id or 0
    model.config.bos_token_id = getattr(tokenizer, "bos_token_id", 1)
    model.config.eos_token_id = getattr(tokenizer, "eos_token_id", 2)

    # Essai d'activer FlashAttention 2 si dispo
    try:
        model.config.attn_implementation = "flash_attention_2"
        print("[FA2] FlashAttention 2 activé")
    except Exception:
        print("[FA2] non disponible — fallback par défaut")

    # LoRA
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=[x.strip() for x in args.lora_targets.split(",") if x.strip()],
        lora_dropout=args.lora_dropout, bias="none", task_type=TaskType.SEQ_CLS
    )
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable()

    # Affichage device du modèle
    try:
        pdev = next(model.parameters()).device
        print("[Model device]", pdev)
    except Exception as e:
        print("[Model device] unknown:", e)

    # Réinit head si demandé (et pas en reprise)
    if args.reinit_head and not args.resume_from:
        _ = reinit_classifier_head(model)

    # Sanity test : geler la base
    if args.freeze_base_for_sanity:
        for n, p in model.named_parameters():
            if not any(k in n for k in ["classifier", "score"]):
                p.requires_grad = False
        print("[Sanity] Base gelée, seul le head est entraîné.")

    # Reprise éventuelle
    if args.resume_from:
        ckpt = os.path.join(args.resume_from, "pytorch_model.bin")
        if not os.path.exists(ckpt):
            ckpt = os.path.join(args.resume_from, "adapter_model.bin")
        if os.path.exists(ckpt):
            print(f"[Resume] {ckpt}")
            adapters_weights = torch.load(ckpt, map_location="cpu")
            try:
                from peft import set_peft_model_state_dict
                set_peft_model_state_dict(model, adapters_weights)
            except Exception:
                model.load_state_dict(adapters_weights, strict=False)
        else:
            print("[Resume] checkpoint introuvable")

    model.print_trainable_parameters()

    # ===================== HF datasets & tokenisation (robuste) =====================
    # Détection de la colonne texte
    text_col = args.text_column
    if text_col is None:
        # Essaye 'text', puis 'note', sinon erreur explicite
        possible = [c for c in ["text", "note", "content", "raw_text"] if c in df_train.columns]
        if possible:
            text_col = possible[0]
        else:
            raise KeyError(
                f"Colonne texte introuvable. Colonnes disponibles: {list(df_train.columns)}. "
                f"Passe --text_column NOM_COL si nécessaire."
            )
    if "label" not in df_train.columns:
        raise KeyError(f"La colonne 'label' est requise dans train/val. Colonnes train: {list(df_train.columns)}")

    print(f"[COLS] text_column='{text_col}' | train cols={list(df_train.columns)} | val cols={list(df_val.columns)}")

    hf_train = Dataset.from_pandas(df_train.reset_index(drop=True))
    hf_val   = Dataset.from_pandas(df_val.reset_index(drop=True))

    def preprocess(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            batch[text_col],
            truncation=True,
            max_length=args.cutoff_len,
        )

    tokenized_train = hf_train.map(preprocess, batched=True)
    tokenized_val   = hf_val.map(preprocess,   batched=True)

    if "label" in tokenized_train.column_names:
        tokenized_train = tokenized_train.rename_column("label", "labels")
    if "label" in tokenized_val.column_names:
        tokenized_val = tokenized_val.rename_column("label", "labels")

    keep = {"input_ids", "attention_mask", "labels"}
    drop_cols_train = [c for c in tokenized_train.column_names if c not in keep]
    drop_cols_val   = [c for c in tokenized_val.column_names   if c not in keep]

    if drop_cols_train:
        print("[CLEAN] train drop:", drop_cols_train)
    if drop_cols_val:
        print("[CLEAN]   val drop:", drop_cols_val)

    tokenized_train = tokenized_train.remove_columns(drop_cols_train)
    tokenized_val   = tokenized_val.remove_columns(drop_cols_val)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    # ==============================================================================

    # ================== Poids de classes ==================
    class_counts = np.bincount(tokenized_train["labels"], minlength=num_labels)
    class_weights = (1.0 / np.maximum(class_counts, 1))
    class_weights = class_weights * (len(class_counts) / class_weights.sum())
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)
    alpha_t = torch.tensor(class_weights, dtype=torch.float32)

    # ===== compute_metrics avec masquage + abstention =====
    def compute_metrics_wrap(eval_pred):
        from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, top_k_accuracy_score

        predictions, labels = eval_pred  # logits [N, C], labels [N]
        predictions = np.array(predictions, copy=True)
        labels = np.asarray(labels)
        C = predictions.shape[1]

        # classes présentes en validation → masque
        present = np.unique(labels)
        unseen = np.setdiff1d(np.arange(C), present)
        if unseen.size > 0:
            predictions[:, unseen] = -1e9

        # abstention optionnelle vers __OTHER__
        y_pred = predictions.argmax(axis=1)
        if args.abstain_tau is not None:
            other_id = None
            for k, v in id2label.items():
                if v == "__OTHER__":
                    other_id = k
                    break
            if other_id is not None:
                exps = np.exp(predictions - predictions.max(axis=1, keepdims=True))
                probs = exps / np.clip(exps.sum(axis=1, keepdims=True), 1e-12, None)
                pmax = probs.max(axis=1)
                low_conf = pmax < float(args.abstain_tau)
                if low_conf.any():
                    y_pred[low_conf] = other_id

        # --- DIAG __OTHER__ ---
        other_id = label2id.get("__OTHER__")
        if other_id is not None:
            n_pred_other = int((y_pred == other_id).sum())
            n_true_other = int((labels == other_id).sum())
            ratio_pred_other = n_pred_other / len(y_pred) if len(y_pred) else 0.0
            print(f"[OTHER] prédits={n_pred_other}  réels={n_true_other}  ratio_pred={ratio_pred_other:.3f}")
            try:
                wandb.log({
                    "eval/pred_OTHER": n_pred_other,
                    "eval/true_OTHER": n_true_other,
                    "eval/ratio_pred_OTHER": ratio_pred_other
                })
            except Exception:
                pass
        # ----------------------

        out = {
            "acc": float(accuracy_score(labels, y_pred)),
            "balanced_acc": float(balanced_accuracy_score(labels, y_pred)),
            "macro_f1": float(f1_score(labels, y_pred, average="macro", zero_division=0)),
        }
        try:
            out["top3"] = float(top_k_accuracy_score(labels, predictions, k=3, labels=present))
            if C >= 5:
                out["top5"] = float(top_k_accuracy_score(labels, predictions, k=5, labels=present))
        except Exception:
            pass
        return out

    # ======== Losses custom + scheduler min-LR ========
    from torch.nn import CrossEntropyLoss

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                labels=None
            )
            logits = outputs.get("logits")
            loss_fct = CrossEntropyLoss(
                weight=class_weights_t.to(logits.device),
                label_smoothing=args.label_smoothing
            )
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

        def create_optimizer_and_scheduler(self, num_training_steps):
            super().create_optimizer_and_scheduler(num_training_steps)
            num_warmup_steps = int(self.args.warmup_ratio * num_training_steps)
            base_lr = self.args.learning_rate
            min_lr = args.min_lr
            min_ratio = min_lr / base_lr if base_lr > 0 else 0.0

            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                progress = (current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_ratio + (1 - min_ratio) * cosine

            from torch.optim.lr_scheduler import LambdaLR
            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)

    focal_loss = FocalLoss(alpha=alpha_t, gamma=args.focal_gamma)

    class FocalTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                labels=None
            )
            logits = outputs.get("logits")
            loss = focal_loss(logits, labels.to(logits.device))
            return (loss, outputs) if return_outputs else loss

        def create_optimizer_and_scheduler(self, num_training_steps):
            super().create_optimizer_and_scheduler(num_training_steps)
            num_warmup_steps = int(self.args.warmup_ratio * num_training_steps)
            base_lr = self.args.learning_rate
            min_lr = args.min_lr
            min_ratio = min_lr / base_lr if base_lr > 0 else 0.0

            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                progress = (current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_ratio + (1 - min_ratio) * cosine

            from torch.optim.lr_scheduler import LambdaLR
            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)

    # Oversampling optionnel
    from torch.utils.data import DataLoader, WeightedRandomSampler

    def make_sample_weights(labels, num_labels):
        counts = np.bincount(labels, minlength=num_labels)
        inv = 1.0 / np.maximum(counts, 1)
        return inv[labels]

    BaseTrainer = FocalTrainer if args.loss_type.lower() == "focal" else WeightedTrainer

    class TrainerWithSampler(BaseTrainer):
        def get_train_dataloader(self):
            dataset = self.train_dataset
            labels = np.array(dataset["labels"])
            sample_weights = make_sample_weights(labels, num_labels)
            sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True
            )
            return DataLoader(
                dataset,
                batch_size=self.args.train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

    TrainerCls = TrainerWithSampler if args.use_oversampling else BaseTrainer

    # ================== Args HF ==================
    per_device_eval_bs = args.eval_batch_size or args.batch_size
    args_tr = TrainingArguments(
        output_dir=output_dir,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,  # surchargé par create_optimizer_and_scheduler
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=per_device_eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        # ---- perf GPU ----
        bf16=True,
        fp16=False,
        optim="adamw_torch_fused",
        # -------------------
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        push_to_hub=False,
        remove_unused_columns=True,
        label_names=["labels"],
        ddp_find_unused_parameters=False if ddp else None,
        report_to="wandb" if args.wandb_project else None,
        run_name=run_name if args.wandb_project else None,
        logging_steps=50,
        seed=args.seed,
        data_seed=args.seed,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    model.config.use_cache = False

    # Log hyperparams “visibles”
    print(f"[HP] epochs= {args.epochs} | lr= {args.lr} min_lr= {args.min_lr} | "
          f"batch= {args.batch_size} grad_accum= {args.grad_accum} | eval_batch_size= {per_device_eval_bs} | "
          f"loss_type= {args.loss_type} label_smoothing= {args.label_smoothing} | "
          f"oversampling= {args.use_oversampling} | metric_for_best= {args.metric_for_best_model} | "
          f"greater_is_better= {args.greater_is_better} | early_stop= {args.early_stopping_patience}")

    trainer = TrainerCls(
        model=model,
        args=args_tr,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrap,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )

    trainer.train(resume_from_checkpoint=args.resume_from)
    model.save_pretrained(output_dir)

    # -------- Évaluation directe (post-train) --------
    preds = trainer.predict(tokenized_val)
    logits = torch.tensor(preds.predictions)   # [N, C]
    y_true = np.array(tokenized_val["labels"])

    # Masque classes absentes
    C = logits.shape[1]
    present = np.unique(y_true)
    unseen = np.setdiff1d(np.arange(C), present)
    logits_np = logits.cpu().numpy()
    if unseen.size > 0:
        logits_np[:, unseen] = -1e9

    # Abstention optionnelle
    if args.abstain_tau is not None and "__OTHER__" in label2id:
        probs = torch.softmax(torch.tensor(logits_np), dim=-1)
        pmax, yhat = probs.max(dim=-1)
        y_pred = yhat.cpu().numpy()
        other_id = label2id["__OTHER__"]
        y_pred = np.where(pmax.cpu().numpy() < args.abstain_tau, other_id, y_pred)
    else:
        y_pred = np.argmax(logits_np, axis=1)

    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, top_k_accuracy_score
    metrics = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    try:
        metrics["top3"] = float(top_k_accuracy_score(y_true, logits_np, k=3, labels=present))
        if num_labels >= 5:
            metrics["top5"] = float(top_k_accuracy_score(y_true, logits_np, k=5, labels=present))
    except Exception:
        pass

    if "__OTHER__" in label2id:
        other_id = label2id["__OTHER__"]
        mask_freq = (y_true != other_id)
        if mask_freq.sum() > 0:
            metrics["macro_f1_frequent_only"] = float(
                f1_score(y_true[mask_freq], y_pred[mask_freq], average="macro")
            )
        metrics["abstention_rate"] = float(np.mean(y_pred == other_id))
        true_other_mask = (y_true == other_id)
        if true_other_mask.sum() > 0:
            metrics["other_recall"] = float(np.mean(y_pred[true_other_mask] == other_id))

    with open(os.path.join(output_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[Eval] Métriques → eval_metrics.json : {metrics}")
    print(f"[NEXT] Évalue à part : python3 ./scripts/eval_drg.py --data_dir {os.path.join(args.data_dir,'val_split')} "
          f"--model_dir {output_dir} --base_model {args.base_model} --cutoff_len {args.cutoff_len}")


if __name__ == "__main__":
    main()

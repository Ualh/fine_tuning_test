from pathlib import Path

from src.training.sft_trainer import SFTTrainerRunner


def _count_records(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def test_load_jsonl_splits_handles_full_alpaca_dataset():
    data_dir = Path(__file__).resolve().parents[1] / "prepared" / "alpaca_full_en"
    splits = SFTTrainerRunner._load_jsonl_splits(data_dir)

    assert set(splits.keys()) == {"train", "validation"}

    expected_counts = {
        "train": _count_records(data_dir / "train.jsonl"),
        "validation": _count_records(data_dir / "val.jsonl"),
    }

    for split_name, expected_count in expected_counts.items():
        dataset = splits[split_name]
        assert len(dataset) == expected_count
        assert dataset.features["text"].dtype == "string"
        sample_text = dataset[0]["text"]
        assert isinstance(sample_text, str)
        assert sample_text

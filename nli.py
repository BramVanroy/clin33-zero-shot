from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Assumes the dataset is on https://hf.co/datasets
DATASET = "dbrd"
DATASET_SPLIT = "test"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

# Verify in the test dataset which index corresponds with which label!
LABELS2IDX = {
    "negatief": 0,
    "positief": 1,
}
F1_AVERAGE = "macro"

MODEL_NAMES = (
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "LoicDL/bert-base-dutch-cased-finetuned-snli",
    "LoicDL/robbert-v2-dutch-finetuned-snli",
    "LoicDL/robbertje-dutch-finetuned-snli",
)
TEMPLATE = "Deze recensie is {}."


def process_dataset(dataset: Dataset):
    true_label_idxs = dataset[LABEL_COLUMN]
    pdout = Path("results/nli")
    pdout.mkdir(exist_ok=True, parents=True)
    with pdout.joinpath("nli_zero_shot_results.txt").open("a", encoding="utf-8") as fhout:
        for model_name in MODEL_NAMES:
            if model_name == "LoicDL/robbertje-dutch-finetuned-snli":
                # Some issues with tokenizer max length, so hard-code it
                config = AutoConfig.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer.model_max_length = config.max_position_embeddings - 2
                pipe = pipeline("zero-shot-classification", model=model_name, tokenizer=tokenizer, device=DEVICE)
            else:
                pipe = pipeline("zero-shot-classification", model=model_name, device=DEVICE)

            pred_idxs = []
            # Get model predictions for the given dataset
            reviews = KeyDataset(dataset, TEXT_COLUMN)
            for pred_dict in tqdm(
                pipe(reviews, candidate_labels=list(LABELS2IDX.keys()), hypothesis_template=TEMPLATE),
                total=len(dataset),
            ):
                # Labels are sorted by likelihood, so first item is most likely
                pred_label = pred_dict["labels"][0]
                pred_idx = LABELS2IDX[pred_label]
                pred_idxs.append(pred_idx)

            f1 = f1_score(true_label_idxs, pred_idxs, average=F1_AVERAGE)
            result_str = f"F1 ({F1_AVERAGE}) score on {DATASET} with {model_name}: {f1:.4f}"
            clf_report = classification_report(true_label_idxs, pred_idxs, target_names=LABELS2IDX.keys(), digits=4)

            print(model_name, f"({pipe.model.num_parameters()} parameters)")
            print(result_str)
            print(clf_report)
            fhout.write(result_str + "\n")
            fhout.write(clf_report + "\n\n")
            fhout.flush()

            ppreds = pdout.joinpath(f"nli-{DATASET.replace('/', '_')}-{model_name.replace('/', '_')}-predictions.txt")

            df = pd.DataFrame(
                [
                    {"review": review, "prediction": pred, "label": label}
                    for review, pred, label in zip(reviews, pred_idxs, true_label_idxs)
                ]
            )
            df.to_csv(ppreds, sep="\t", encoding="utf-8", index=False)


def get_dataset():
    dataset = load_dataset(DATASET, split=DATASET_SPLIT)

    print(f"DATASET SIZE: {len(dataset):,}")
    for label, label_idx in LABELS2IDX.items():
        print(f"Dataset no. occurrences for {label}: {dataset[LABEL_COLUMN].count(label_idx):,}")

    return dataset


def main():
    dataset = get_dataset()
    process_dataset(dataset)


if __name__ == "__main__":
    main()

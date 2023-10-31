import os
from pathlib import Path
from time import sleep

import openai
import pandas as pd
from datasets import Dataset, load_dataset
from outlines import models
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

# OpenAI API key taken from env
openai.api_key = os.getenv("OPENAI_API_KEY")

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
TEMPERATURE = 1.0  # API default
MAX_TOKENS = 20

USE_TMP_FILE = True
MODEL_NAME = "gpt-3.5-turbo"

PROMPT = "Is het sentiment in de volgende Nederlandstalige boekrecensie positief of negatief?"


def process_dataset(dataset: Dataset):
    true_label_idxs = dataset[LABEL_COLUMN]
    pdout = Path("results/prompt")
    pdout.mkdir(exist_ok=True, parents=True)
    with pdout.joinpath("openai_zero_shot_results.txt").open("a", encoding="utf-8") as fhout:
        model = models.openai.OpenAICompletion(
            MODEL_NAME,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        pred_idxs = []
        reviews = KeyDataset(dataset, TEXT_COLUMN)

        pftmp = Path("tmp-openai-results.txt")
        if pftmp.exists() and USE_TMP_FILE:
            pred_idxs = list(map(int, pftmp.read_text(encoding="utf-8").splitlines()))

        with pftmp.open("w", encoding="utf-8") as fhtemp:
            skip_to = len(pred_idxs)
            if pred_idxs:
                fhtemp.write("\n".join(map(str, pred_idxs)) + "\n")
            for review_idx, review in tqdm(enumerate(reviews), total=len(reviews), desc=MODEL_NAME):
                if review_idx < skip_to:
                    continue
                prompted_review = f"{PROMPT}\n\n{review}"
                num_retries = 3
                while num_retries:
                    try:
                        pred_label = model(prompt=prompted_review, is_in=list(LABELS2IDX.keys())).replace(" ", "")
                    except openai.error.OpenAIError as exc:
                        num_retries -= 1
                        sleep(30)
                        # Handle API error here, e.g. retry or log
                        print(
                            f"OpenAI API returned an API Error ({num_retries} retries remaining for text #{review_idx:,}): {exc}"
                        )
                        continue
                    else:
                        pred_idx = LABELS2IDX[pred_label]
                        pred_idxs.append(pred_idx)
                        fhtemp.write(f"{pred_idx}\n")
                        fhtemp.flush()
                        break

                if not num_retries:
                    raise openai.error.OpenAIError(
                        "OpenAI API threw three errors sequentially (see above) so we're"
                        " giving up. Temporary results are saved to the temp file. Feel"
                        " free to restart the script to try again where you left off."
                    )
        # If we have gotten this far, we can delete the tmp file
        pftmp.unlink()
        f1 = f1_score(true_label_idxs, pred_idxs, average=F1_AVERAGE, labels=list(LABELS2IDX.values()))
        result_str = f"F1 ({F1_AVERAGE}) score on {DATASET} with {MODEL_NAME}: {f1:.4f}"
        clf_report = classification_report(
            true_label_idxs, pred_idxs, target_names=LABELS2IDX.keys(), labels=list(LABELS2IDX.values()), digits=4
        )

        print(MODEL_NAME)
        print(result_str)
        print(clf_report)
        fhout.write(result_str + "\n")
        fhout.write(clf_report + "\n\n")
        fhout.flush()

    ppreds = pdout.joinpath(f"openai-{DATASET.replace('/', '_')}-{MODEL_NAME.replace('/', '_')}-predictions.txt")

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

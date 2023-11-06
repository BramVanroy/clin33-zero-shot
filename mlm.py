from pathlib import Path
import sys
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent))

from logits_fill_mask import LogitsFillMaskPipeline
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.pipelines.pt_utils import KeyDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    "FremyCompany/roberta-large-nl-oscar23",
    "pdelobelle/robbert-v2-dutch-base",
    "GroNLP/bert-base-dutch-cased",
    "DTAI-KULeuven/robbertje-1-gb-merged",
)
MODEL_ADD_PREFIX_SPACE = {
    "FremyCompany/roberta-large-nl-oscar23": "Ġ",
    "pdelobelle/robbert-v2-dutch-base": "Ġ",
    "GroNLP/bert-base-dutch-cased": None,
    "DTAI-KULeuven/robbertje-1-gb-merged": "Ġ",
}
TEMPLATE = "De volgende recensie is{}."
USE_RESCALING = (True, False)


@torch.inference_mode
def get_baseline_logits(model, tokenizer, mask_prompt: str, labels: List[str]) -> Dict[str, float]:
    encoded = tokenizer(mask_prompt, return_tensors="pt").to(model.device)
    outputs = model(**encoded)["logits"].cpu()

    masked_index = torch.nonzero(encoded.input_ids[0].cpu() == tokenizer.mask_token_id, as_tuple=False).item()
    logits = outputs[0, masked_index, :]

    baseline_logits = {}
    for label in labels:
        label_token_id = tokenizer.convert_tokens_to_ids(label)
        baseline_logits[label] = logits[label_token_id].item()
    return baseline_logits


@torch.inference_mode
def process_dataset(dataset: Dataset):
    true_label_idxs = dataset[LABEL_COLUMN]
    pdout = Path("results/mlm")
    pdout.mkdir(exist_ok=True, parents=True)
    with pdout.joinpath("mlm_zero_shot_results.txt").open("w", encoding="utf-8") as fhout:
        for model_name in MODEL_NAMES:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if "robbertje" in model_name.lower():
                # Some issues with tokenizer max length, so hard-code it
                tokenizer.model_max_length = 512

            # Is this a tokenizer that includes prefix spaces in tokens? If so, add "space" to token
            has_space_in_token = MODEL_ADD_PREFIX_SPACE[model_name] is not None
            labels_maybe_with_prefix = [
                f"{MODEL_ADD_PREFIX_SPACE[model_name]}{label}" if has_space_in_token else label for label in LABELS2IDX
            ]
            labels_no_prefix = list(LABELS2IDX.keys())
            # Make sure that the labels (maybe with prefix space) exist in the vocabulary
            if not all(label in tokenizer.vocab for label in labels_maybe_with_prefix):
                raise ValueError(
                    "All labels must be in the vocabulary."
                    " Perhaps you did not correctly specify the prefix space (if any)"
                )

            # If a prefix space is present in the token, we do not need to add it to the text separately
            mask_prompt = (
                TEMPLATE.format(tokenizer.mask_token)
                if has_space_in_token
                else TEMPLATE.format(f" {tokenizer.mask_token}")
            )
            model = AutoModelForMaskedLM.from_pretrained(model_name)
            model = model.to(DEVICE)
            model.eval()

            for use_rescaling in USE_RESCALING:
                pipe = LogitsFillMaskPipeline(
                    model=model,
                    tokenizer=tokenizer,
                    device=DEVICE,
                    targets=labels_maybe_with_prefix,
                    tokenizer_kwargs={"truncation": True},
                    top_k=2,
                    return_logits=use_rescaling,
                )

                baseline_logits = {}
                if use_rescaling:
                    baseline_logits = get_baseline_logits(model, tokenizer, mask_prompt, labels_maybe_with_prefix)

                reviews = KeyDataset(dataset, TEXT_COLUMN)
                # Review at the end so that when truncated the mask is not cut off
                reviews = [f"{mask_prompt}\n{review}" for review in reviews]
                pred_idxs = []
                # pred_scores: A list of dictionaries with keys "score", "token" (ID), "tokenstr", "sequence"
                # but restricted to only the tokens that we are interested in ("targets" in pipeline creation)
                for pred_scores in tqdm(pipe(reviews), total=len(reviews)):
                    if use_rescaling:
                        rescaled_logits = []
                        # Loop to ensure order
                        for label in labels_maybe_with_prefix:
                            baseline = baseline_logits[label]
                            label_token_id = tokenizer.convert_tokens_to_ids(label)
                            orig_logits = next(
                                pred["logits"] for pred in pred_scores if pred["token"] == label_token_id
                            )
                            rescaled_logits.append(baseline - orig_logits)
                        probs = torch.tensor(rescaled_logits).softmax(dim=-1)
                        pred_idx = probs.topk(1).indices[0].item()
                        pred_label_idx = labels_no_prefix[pred_idx]
                    else:
                        pred_scores = sorted(pred_scores, key=lambda item: item["score"], reverse=True)
                        # 'token_str' does not have the prefix special characters but has those replaced by regular spaces
                        # so we still have to strep them away
                        pred_label_idx = pred_scores[0]["token_str"].strip()

                    pred_idx = LABELS2IDX[pred_label_idx]
                    pred_idxs.append(pred_idx)

                f1 = f1_score(true_label_idxs, pred_idxs, average=F1_AVERAGE)
                result_str = (
                    f"F1 ({F1_AVERAGE}) score on {DATASET} with {model_name} (rescale={use_rescaling}): {f1:.4f}"
                )
                clf_report = classification_report(
                    true_label_idxs, pred_idxs, target_names=LABELS2IDX.keys(), digits=4
                )

                print(model_name, f"({pipe.model.num_parameters()} parameters)")
                print(result_str)
                print(clf_report)
                fhout.write(result_str + "\n")
                fhout.write(clf_report + "\n\n")
                fhout.flush()

                ppreds = pdout.joinpath(
                    f"mlm-{DATASET.replace('/', '_')}"
                    f"-{model_name.replace('/', '_')}"
                    f"-{'rescaled-' if use_rescaling else ''}"
                    f"predictions.txt"
                )

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

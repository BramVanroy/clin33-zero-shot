import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset
from outlines import models
from outlines.text import generate
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from transformers import BitsAndBytesConfig
from transformers.pipelines.pt_utils import KeyDataset

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


class PromptTemplate(StrEnum):
    LLAMA2_VANROY = """[INST] <<SYS>>
Je bent een behulpzame, respectvolle en eerlijke assistent. Antwoord altijd zo behulpzaam mogelijk. Je antwoorden mogen geen schadelijke, onethische, racistische, seksistische, gevaarlijke of illegale inhoud bevatten. Zorg ervoor dat je antwoorden sociaal onbevooroordeeld en positief van aard zijn.

Als een vraag nergens op slaat of feitelijk niet coherent is, leg dan uit waarom in plaats van iets niet correct te antwoorden. Als je het antwoord op een vraag niet weet, deel dan geen onjuiste informatie.
<</SYS>>

{prompt} [/INST] """

    LLAMA2 = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{prompt} [/INST] """

    FALCON_VANROY = """Hieronder staat een instructie `Instruction` die een taak beschrijft, gecombineerd met een invoer `Input` die verdere context biedt. Schrijf een antwoord na `Response:` dat het verzoek op de juiste manier voltooit of beantwoordt.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    # Falcon was finetuned on BAIZE data https://huggingface.co/tiiuae/falcon-40b-instruct/discussions/1
    FALCON = """The conversation between human and AI assistant.
[|Human|] {prompt}
[|AI|] 
"""


MODEL_NAMES = (
    "meta-llama/Llama-2-7b-chat-hf",
    # "BramVanroy/Llama-2-13b-chat-dutch",
    # "meta-llama/Llama-2-13b-chat-hf",
    # "BramVanroy/falcon-40b-ft-alpaca-dolly-dutch",
    # "tiiuae/falcon-40b-instruct",
)
MODEL_NAME2PROMPT_TEPLATE = {
    "meta-llama/Llama-2-7b-chat-hf": PromptTemplate.LLAMA2,
    "BramVanroy/Llama-2-13b-chat-dutch": PromptTemplate.LLAMA2_VANROY,
    "meta-llama/Llama-2-13b-chat-hf": PromptTemplate.LLAMA2,
    "BramVanroy/falcon-40b-ft-alpaca-dolly-dutch": PromptTemplate.FALCON_VANROY,
    "tiiuae/falcon-40b-instruct": PromptTemplate.FALCON,
}
PROMPT = "Is het sentiment in de volgende Nederlandstalige boekrecensie positief of negatief?"
USE_TEMPLATE = (True, False)


def process_dataset(dataset: Dataset):
    true_label_idxs = dataset[LABEL_COLUMN]
    pdout = Path("results/prompt")
    pdout.mkdir(exist_ok=True, parents=True)
    with pdout.joinpath("prompt_zero_shot_results.txt").open("a", encoding="utf-8") as fhout:
        for model_name in MODEL_NAMES:
            prompt_tmpl = MODEL_NAME2PROMPT_TEPLATE[model_name]
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="bfloat16",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs = {"quantization_config": bnb_config, "trust_remote_code": True}
            model = models.transformers(model_name, model_kwargs=model_kwargs, device="auto")

            for use_template in USE_TEMPLATE:
                pred_idxs = []
                reviews = KeyDataset(dataset, TEXT_COLUMN)
                for review in tqdm(reviews, total=len(reviews), desc=model_name):
                    if use_template:
                        if prompt_tmpl == PromptTemplate.FALCON_VANROY:
                            prompted_review = prompt_tmpl.format(instruction=PROMPT, input=review)
                        elif prompt_tmpl in (PromptTemplate.LLAMA2_VANROY, PromptTemplate.LLAMA2):
                            prompted_review = prompt_tmpl.format(prompt=f"{PROMPT}\n\n{review}")
                        elif prompt_tmpl == PromptTemplate.FALCON:
                            prompted_review = prompt_tmpl.format(prompt=f"{PROMPT}\n{review}")
                    else:
                        prompted_review = f"{PROMPT}\n\n{review}"

                    pred_label = generate.choice(model, list(LABELS2IDX.keys()))(prompted_review).replace(" ", "")
                    pred_idx = LABELS2IDX[pred_label]
                    pred_idxs.append(pred_idx)

                f1 = f1_score(true_label_idxs, pred_idxs, average=F1_AVERAGE, labels=list(LABELS2IDX.values()))
                result_str = (
                    f"F1 ({F1_AVERAGE}) score on {DATASET} with {model_name} (template={use_template}): {f1:.4f}"
                )
                clf_report = classification_report(
                    true_label_idxs,
                    pred_idxs,
                    target_names=LABELS2IDX.keys(),
                    labels=list(LABELS2IDX.values()),
                    digits=4,
                )

                print(model_name)
                print(result_str)
                print(clf_report)
                fhout.write(result_str + "\n")
                fhout.write(clf_report + "\n\n")
                fhout.flush()

                ppreds = pdout.joinpath(
                    f"prompt-{DATASET.replace('/', '_')}-{model_name.replace('/', '_')}-{'' if use_template else 'no_template-'}predictions.txt"
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

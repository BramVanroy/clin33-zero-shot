# Zero Shot Classification Methods

**Note this code base is not pretty or well documented. It's not user-friendly (change arguments in the files instead of from CLI), but it is what it is. Use at your own risk.**

Code to do zero-shot classification on datasets from the HuggingFace Hub with different methodologies. Supports:

- NLI (via HF zero-shot pipeline)
- MLM (via HF fill-mask pipeline)
- Generative AI (via {transformers,openai}+outlines)

## Installation

Requires Python 3.8 or higher (tested with Python 3.10).

```shell
python -m pip install -r requirements.txt
```

## Usage

Just run the scripts individually. They are completely standalone and can be used separately.

As said before, these scripts were written in a quick-and-dirty manner. Any adaptations you want to do need to happen
in code. The most important variables are constants at the top of the file for you to change.

Note that all available GPUs will be used at the same time for generation models only. For the others, one GPU
will be used if CUDA is available, otherwise CPU.

For the openai usage, make sure that your OpenAI key is set as an environment variable `OPENAI_API_KEY`. I found that 
generation works very slowly and often leads to timeouts. Not sure whether that is related to the things that `outlines`
is doing under the hood or not. Intermediate process is being saved so if the process crashes due to a timeout or 
other errors, you should be able to just restart and it will continue where it left off. This is not ideal but that's
how it is for now. (If you have the time, it is worthwhile to reimplement this querying with multiprocessing to greatly
speed up the process.)

## Citation

If you use this code, please make sure to provide correct attribution. Code is licensed under GPLv3. If you use the code,
it would also be kind to cite our upcoming work. I'll add needed citation information here when that work is available.

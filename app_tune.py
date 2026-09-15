#!/usr/bin/env python3
"""app_tune.py - fine tunes the checkpoint with TRL, and repacks the result.

The engine reads the Hugging Face checkpoint directly and has no trainer of
its own, so a tune happens on the reference side: TRL's `SFTTrainer` trains a
LoRA adapter over the frozen base weights, and the adapter is then folded back
into the float checkpoint, which the engine already reads without conversion.
The tune is a workflow helper like the other `.py` files, not part of the
engine.

Two steps, run separately because the second needs only the first's output:

    python3 app_tune.py --model model --train
    python3 app_tune.py --model model --merge

`--train` writes `build/tune/adapter` (the PEFT adapter) and `build/tune/`
(its trainer state). `--merge` reads the adapter, folds it into the base
weights, and writes `build/tune/merged`, a checkpoint in the same layout as
`model/` that the engine and the reference both read directly:

    python3 run.py run -- generate --model build/tune/merged --prompt "Hello!"

The dataset is `data_tune.jsonl` beside this script: one JSON object per line
with `prompt` and `completion`. It is a minimal example to be extended; add
lines, do not restructure it, because the loader reads exactly those two keys.

The Liquid stack has two operators, and the tune trains the attention one
only. The short convolution layers stay frozen: their depthwise `conv` is not
a linear layer a LoRA adapter attaches to, and a text tune has no reason to
move them. The feed forward stays frozen with them, so the adapter reaches
the `q_proj`, `k_proj`, `v_proj`, and `out_proj` of the attention layers and
nothing else.

When torch, transformers, peft or trl are missing, the script reports what it
skipped and exits zero, so it stays usable inside a build pipeline.
"""

import argparse
import json
import os
import sys

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
TUNE_PATH = os.path.join(ROOT_PATH, "build", "tune")
ADAPTER_PATH = os.path.join(TUNE_PATH, "adapter")
MERGED_PATH = os.path.join(TUNE_PATH, "merged")
DATA_PATH = os.path.join(ROOT_PATH, "data_tune.jsonl")

# Small on purpose: the example dataset is a dozen lines, and a tune that
# overruns it teaches nothing but takes minutes. Raise these with the data.
TRAIN_EPOCHS = 3
TRAIN_BATCH = 1
TRAIN_RATE = 2e-4
TRAIN_CUT = 256
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROP = 0.05

# The attention layers only. `self_attn` pins the match to them, which keeps
# the convolution layers' own `out_proj` frozen beside the rest of the conv
# stack. PEFT reads a string target as a regular expression.
TARGET_PATTERN = r".*\.self_attn\.(q_proj|k_proj|v_proj|out_proj)"


def need_modules():
    """Imports what a tune needs, or prints what is missing and returns None."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import peft  # noqa: F401
        import trl  # noqa: F401
    except ImportError as miss:
        print("skip: %s; pip install torch transformers peft trl datasets" % miss)
        return None
    return True


def load_rows(data_path):
    """Reads the JSONL dataset, one prompt/completion pair per line."""
    row_list = []
    with open(data_path, "r", encoding="utf-8") as hand:
        for number, line in enumerate(hand, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "prompt" not in row or "completion" not in row:
                raise ValueError("%s line %d: want prompt and completion" % (data_path, number))
            row_list.append({"prompt": row["prompt"], "completion": row["completion"]})
    if not row_list:
        raise ValueError("%s holds no rows" % data_path)
    return row_list


def train_step(model_path, out_path, data_path, epochs, batch, rate, cut):
    """Trains a LoRA adapter over the frozen base with TRL's SFTTrainer."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    book = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto"
    )
    model.config.use_cache = False

    row_list = load_rows(data_path)
    # The chat template decides the shape, so the tune trains on what the
    # engine is actually asked to serve rather than on raw concatenations.
    def frame(row):
        turn_list = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["completion"]},
        ]
        return book.apply_chat_template(turn_list, tokenize=False)

    data = Dataset.from_list([{"text": frame(row)} for row in row_list])

    setting = SFTConfig(
        output_dir=out_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=4,
        learning_rate=rate,
        max_length=cut,
        packing=False,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
    )
    adapter = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROP,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_PATTERN,
    )
    trainer = SFTTrainer(
        model=model,
        args=setting,
        train_dataset=data,
        processing_class=book,
        peft_config=adapter,
    )
    trainer.train()
    trainer.save_model(os.path.join(out_path, "adapter"))
    book.save_pretrained(os.path.join(out_path, "adapter"))
    return os.path.join(out_path, "adapter")


def merge_step(model_path, adapter_path, out_path):
    """Folds the adapter into the base weights and writes a plain checkpoint."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(out_path)
    book = AutoTokenizer.from_pretrained(model_path)
    book.save_pretrained(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="inferliqu fine tuning with TRL")
    parser.add_argument("--model", default=os.environ.get("INFERLIQU_MODEL", "model"),
                        help="checkpoint folder in huggingface layout")
    parser.add_argument("--data", default=DATA_PATH,
                        help="JSONL dataset with prompt and completion per line")
    parser.add_argument("--train", action="store_true",
                        help="train a LoRA adapter into %s" % ADAPTER_PATH)
    parser.add_argument("--merge", action="store_true",
                        help="fold the adapter into %s" % MERGED_PATH)
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--batch", type=int, default=TRAIN_BATCH)
    parser.add_argument("--rate", type=float, default=TRAIN_RATE)
    parser.add_argument("--cut", type=int, default=TRAIN_CUT)
    flag = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    if not flag.train and not flag.merge:
        print("nothing to do: pass --train, --merge, or both")
        return 0
    if not need_modules():
        return 0
    if flag.train:
        if not flag.model or not os.path.isdir(flag.model):
            print("skip: no checkpoint folder given; pass --model or set INFERLIQU_MODEL")
            return 0
        if not os.path.isfile(flag.data):
            print("skip: no dataset at %s" % flag.data)
            return 0
        made = train_step(flag.model, TUNE_PATH, flag.data,
                          flag.epochs, flag.batch, flag.rate, flag.cut)
        print("adapter written to %s" % made)
    if flag.merge:
        if not os.path.isdir(ADAPTER_PATH):
            print("skip: no adapter at %s; run --train first" % ADAPTER_PATH)
            return 0
        made = merge_step(flag.model, ADAPTER_PATH, MERGED_PATH)
        print("merged checkpoint written to %s" % made)
    return 0


if __name__ == "__main__":
    sys.exit(main())

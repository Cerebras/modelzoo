# Copyright 2022 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import random
import re
import time
import unicodedata
from multiprocessing import Pool, cpu_count

import fasttext
import yaml
from datasets import load_dataset
from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm


def parse_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def normalization(text: str) -> str:
    tokens = wordpunct_tokenize(text)
    processed_tokens = []
    for token in tokens:
        token = token.lower()
        if token.isdigit():
            processed_tokens.append("<NUM>")
        elif len(token) <= 100:
            processed_tokens.append(token)
    preprocessed_text = " ".join(processed_tokens)
    preprocessed_text = re.sub(r"[\n\r]+", " ", preprocessed_text)
    preprocessed_text = re.sub(r"[-_]+", " ", preprocessed_text)
    preprocessed_text = re.sub(r"[^a-zA-Z0-9\s<NUM>]", "", preprocessed_text)
    preprocessed_text = re.sub(r"\s+", " ", preprocessed_text).strip()
    return preprocessed_text


def preprocess_for_fasttext(text: str) -> str:
    if isinstance(text, (bytes, bytearray)):
        text = text.decode("utf-8")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s", " ", text)
    text = text.replace("\n", " <EOS> ")
    text = re.sub(r"\s+", " ", text)
    text = normalization(text)
    MAX_LINE_SIZE = 1024
    lines = text.split("<EOS>")
    processed_lines = []
    for line in lines:
        tokens = line.split()
        if len(tokens) > MAX_LINE_SIZE:
            processed_lines.extend(
                [
                    " ".join(tokens[i : i + MAX_LINE_SIZE])
                    for i in range(0, len(tokens), MAX_LINE_SIZE)
                ]
            )
        else:
            processed_lines.append(line)
    return " <EOS> ".join(processed_lines).strip()


def evaluate_model(model, file_path, split_name):
    t1 = time.time()
    n, prec, rec = model.test(file_path)
    t2 = time.time()

    print(f"⏱ Evaluation took {t2 - t1:.1f}s on dataset split - {split_name}")
    print(f"  Total samples: {n}")
    print(f"  Precision:          {prec:.4f}")
    print(f"  Recall:             {rec:.4f}")
    print(f"  Approx. Accuracy:   {((prec + rec) / 2):.4f}")
    print(f"\nTotal runtime: {t2 - t1:.1f}s")


def _worker_process_stream_sample(args):
    ex, is_math = args
    text = preprocess_for_fasttext(ex["text"])
    label = "MATH" if is_math else "NON_MATH"
    return f"__label__{label} {text}\n", is_math


def stream_to_files(cfg, train_path, val_path, math_iter, non_math_iter):
    total_target = cfg["training"]["total_samples"]
    math_count, non_math_count = 0, 0

    # truncate files
    open(train_path, "w", encoding="utf-8").close()
    open(val_path, "w", encoding="utf-8").close()

    train_prob = cfg["training"]["train_prob"]
    total = 2 * total_target

    pbar = tqdm(total=total, desc="Creating initial dataset", unit="samples")
    pbar.set_postfix({"math": math_count, "non-math": non_math_count})

    # build a simple generator of raw examples
    def example_generator():
        while math_count < total_target or non_math_count < total_target:
            pick_math = math_count < total_target and (
                non_math_count >= total_target or random.random() < 0.5
            )
            try:
                ex = next(math_iter) if pick_math else next(non_math_iter)
            except StopIteration:
                # if one runs out, force the other
                pick_math = not pick_math
                try:
                    ex = next(math_iter) if pick_math else next(non_math_iter)
                except StopIteration:
                    break
            yield ex, pick_math

    # start pool
    workers = max(1, cpu_count() - 1)
    pool = Pool(processes=workers)

    # Use context managers for proper file handling
    with (
        open(train_path, "a", encoding="utf-8") as train_file,
        open(val_path, "a", encoding="utf-8") as val_file,
    ):

        # feed into pool in moderately‐sized batches
        for result, was_math in pool.imap_unordered(
            _worker_process_stream_sample, example_generator(), chunksize=256
        ):
            # update counts
            if was_math:
                math_count += 1
            else:
                non_math_count += 1

            # write to train or val
            if random.random() < train_prob:
                train_file.write(result)
            else:
                val_file.write(result)

            # progress
            pbar.update(1)
            pbar.set_postfix({"math": math_count, "non-math": non_math_count})

            if math_count >= total_target and non_math_count >= total_target:
                break

    # cleanup
    pbar.close()
    pool.close()
    pool.join()

    print(f"✅ Final counts – math: {math_count}, non-math: {non_math_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()
    cfg = parse_config(args.config)

    random.seed(cfg["random_seed"])
    os.makedirs(cfg["output"]["output_dir"], exist_ok=True)

    # Create train and validation directories
    train_path = os.path.join(
        cfg["output"]["output_dir"], cfg["output"]["train_filename"]
    )
    val_path = os.path.join(
        cfg["output"]["output_dir"], cfg["output"]["val_filename"]
    )

    math_config = cfg["datasets"]["math_config"]
    non_math_config = cfg["datasets"]["non_math_config"]
    math_ds_stream = load_dataset(
        cfg["datasets"]["math_dataset_name"],
        name=math_config,
        split="train",
        streaming=True,
        cache_dir=cfg["datasets"]["cache_math"],
    )
    non_math_ds_stream = load_dataset(
        cfg["datasets"]["non_math_dataset_name"],
        name=non_math_config,
        split="train",
        streaming=True,
        cache_dir=cfg["datasets"]["cache_non_math"],
    )
    math_iter = iter(math_ds_stream)
    non_math_iter = iter(non_math_ds_stream)

    start = time.time()
    print("→ Creating Initial Training and Validation sets")
    stream_to_files(cfg, train_path, val_path, math_iter, non_math_iter)
    print(f"✅ Created in {time.time() - start:.1f}s")

    model_path = os.path.join(
        cfg["output"]["output_dir"],
        f"math_classifier.bin",
    )
    t0 = time.time()
    model = fasttext.train_supervised(
        input=train_path, **cfg["training"]["fasttext_params"]
    )
    model.save_model(model_path)
    t1 = time.time()
    print(f"✅ Model trained and saved to {model_path} in {t1 - t0} seconds.")

    print("→ Evaluating on validation set…")
    evaluate_model(model, val_path, "validation")
    print("→ Evaluating on training set…")
    evaluate_model(model, train_path, "training")
    print(f"\nEvaluation took : {time.time() - t1:.1f}s")


if __name__ == "__main__":
    main()

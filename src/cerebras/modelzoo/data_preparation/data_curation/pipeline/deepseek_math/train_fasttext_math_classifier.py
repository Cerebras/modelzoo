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
import shutil
import time
import unicodedata
from multiprocessing import Pool, cpu_count

import fasttext
import yaml
from datasets import load_dataset
from fasttext import load_model
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


def latex_symbol_filter(text: str) -> bool:

    MATH_KEYWORDS = [
        'MathJax',
        'mathjax',
        '<math',
        'math-container',
        'katex.min.css',
        'latex.php',
        'codecogs',
        'tex.cgi',
        'class="tex"',
        "class='tex'",
    ]
    latex_math_commands = [
        "\\end",
        "\\begin",
        "\\ref",
        "\\frac",
        "\\label",
        "\\bf",
        "\\right",
        "\\left",
        "\\rm",
        "\\alpha",
        "\\mu",
        "\\def",
        "\\it",
        "\\pi",
        "\\sigma",
        "\\sum",
        "\\lambda",
        "\\beta",
        "\\nu",
        "\\partial",
        "\\int",
        "\\delta",
        "\\rho",
        "\\phi",
        "\\gamma",
        "\\omega",
        "\\over",
        "\\nonumber",
        "\\bar",
        "\\sqrt",
        "\\theta",
        "\\tau",
        "\\em",
        "\\rangle",
        "\\hat",
        "\\tilde",
        "\\cal",
        "\\hline",
        "\\item",
        "\\psi",
        "\\vec",
        "\\langle",
        "\\epsilon",
        "\\eta",
        "\\cdot",
        "\\in",
        "\\xi",
        "\\infty",
        "\\quad",
        "\\mathcal",
        "\\times",
        "\\emph",
        "\\mathbf",
        "\\prime",
        "\\be",
        "\\mathrm",
        "\\ee",
        "\\vspace",
        "\\pm",
        "\\chi",
        "\\ell",
        "\\text",
        "\\qquad",
        "\\noindent",
        "\\to",
        "\\varphi",
        "\\hspace",
        "\\leq",
        "\\cos",
        "\\eqref",
        "\\overline",
        "\\sin",
        "\\kappa",
        "\\hbox",
        "\\rightarrow",
        "\\varepsilon",
        "\\textit",
        "\\dagger",
        "\\big",
        "\\otimes",
        "\\equiv",
        "\\zeta",
        "\\dot",
        "\\ln",
    ]
    latex_regex = re.compile('\\\\[a-z]{2,}')
    original_regex = re.compile('|'.join(MATH_KEYWORDS))
    if original_regex.search(text):
        return True
    if latex_regex.search(text):
        for term in latex_math_commands:
            if term in text:
                return True
    return False


# this will live in each worker process
_worker_model = None


def _worker_init(model_path):
    """
    Initializer for each worker: load the FastText model only once per process.
    """
    global _worker_model
    import fasttext

    _worker_model = fasttext.load_model(model_path)


def _worker_process_sample(args):
    """
    Wrapper that unpacks the arguments tuple and calls your logic,
    but using the worker-local `_worker_model`.
    """
    ex, is_math = args
    # your pre‐existing preprocess / filter logic:
    text = preprocess_for_fasttext(ex["text"])
    predicted_label, confidence = _worker_model.predict(text)
    predicted_label = predicted_label[0].replace("__label__", "")

    if (is_math and predicted_label == "NON_MATH") or (
        not is_math
        and predicted_label == "MATH"
        and not latex_symbol_filter(ex["text"])
    ):
        return f"__label__{predicted_label} {text}\n", is_math
    else:
        return None, is_math


def augment_dataset(model_path, cfg, iteration_number):
    # --- copy over last‐iteration files ----
    out_dir = cfg["output"]["output_dir"]
    train_base = os.path.join(out_dir, cfg["output"]["train_filename"])
    val_base = os.path.join(out_dir, cfg["output"]["val_filename"])
    curr_train = f"{train_base}/iteration_{iteration_number}.txt"
    curr_val = f"{val_base}/iteration_{iteration_number}.txt"
    prev_train = f"{train_base}/iteration_{iteration_number - 1}.txt"
    prev_val = f"{val_base}/iteration_{iteration_number - 1}.txt"

    os.makedirs(os.path.dirname(curr_train), exist_ok=True)

    shutil.copy(prev_train, curr_train)
    shutil.copy(prev_val, curr_val)

    # --- set up streaming datasets ---
    math_cfg = cfg["datasets"]["math_configs"][0]
    non_math_cfg = cfg["datasets"]["non_math_configs"][0]

    math_stream = load_dataset(
        cfg["datasets"]["math_dataset_name"],
        name=math_cfg,
        split="train",
        streaming=True,
        cache_dir=cfg["datasets"]["cache_math"],
    )
    non_math_stream = load_dataset(
        cfg["datasets"]["non_math_dataset_name"],
        name=non_math_cfg,
        split="train",
        streaming=True,
        cache_dir=cfg["datasets"]["cache_non_math"],
    )

    math_iter = iter(math_stream)
    non_math_iter = iter(non_math_stream)

    total_per_class = cfg["training"]["total_samples"]
    train_prob = cfg["training"]["train_prob"]

    # half the total, because we draw math & non-math each up to total_per_class
    pbar = tqdm(
        total=total_per_class * 2,
        desc=f"Augmenting (iter {iteration_number})",
        unit="samples",
    )
    pbar.set_postfix({"math": 0, "non-math": 0})

    # counters
    counts = {"math": 0, "non-math": 0}
    out_counts = {
        "train": {"math": 0, "non-math": 0},
        "val": {"math": 0, "non-math": 0},
    }

    # prepare multiprocessing pool
    num_workers = max(1, cpu_count() - 1)
    pool = Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(model_path,),
        maxtasksperchild=1000,  # guard against leaks
    )

    def sample_generator():
        """
        Yields (example, is_math_flag) until both classes have reached the target.
        """
        nonlocal total_per_class
        while (
            counts["math"] < total_per_class
            or counts["non-math"] < total_per_class
        ):
            pick_math = random.random() < 0.5
            # force pick if one is exhausted
            if counts["math"] >= total_per_class:
                pick_math = False
            elif counts["non-math"] >= total_per_class:
                pick_math = True

            try:
                ex = next(math_iter) if pick_math else next(non_math_iter)
            except StopIteration:
                # if one stream runs dry, reduce targets to what we've got so far
                reached = counts["math"] if pick_math else counts["non-math"]
                total_adj = reached
                total_per_class = total_adj
                pbar.total = total_adj * 2
                continue

            yield ex, pick_math

    # map the generator into the pool, in chunks
    # we use imap_unordered to get results as soon as they're ready
    for result, is_math in pool.imap_unordered(
        _worker_process_sample,
        sample_generator(),
        chunksize=128,  # tune this for your system
    ):

        if result is None:
            continue

        label = "math" if is_math else "non-math"
        counts[label] += 1

        # choose train vs val
        split = "train" if random.random() < train_prob else "val"
        out_counts[split][label] += 1

        # write immediately
        target_f = curr_train if split == "train" else curr_val
        with open(target_f, "a", encoding="utf-8") as fout:
            fout.write(result)

        pbar.update(1)
        pbar.set_postfix(
            {"math": counts["math"], "non-math": counts["non-math"]}
        )

        # break if done
        if (
            counts["math"] >= total_per_class
            and counts["non-math"] >= total_per_class
        ):
            break

    pbar.close()
    pool.close()
    pool.join()

    print(
        f"Train: math={out_counts['train']['math']}, non-math={out_counts['train']['non-math']}\n"
        f" Val:  math={out_counts['val']['math']}, non-math={out_counts['val']['non-math']}"
    )


def process_stream_sample(ex, is_math):
    text = preprocess_for_fasttext(ex["text"])
    label = "MATH" if is_math else "NON_MATH"
    return f"__label__{label} {text}\n", is_math


def evaluate_model(model, file_path, split_name):

    t1 = time.time()
    n, prec, rec = model.test(file_path)
    t2 = time.time()

    print(f"⏱ Evaluation took {t2 - t1:.1f}s on dataset split - {split_name}")
    print(f"  Total samples: {n}")
    print(f"  Precision:          {prec:.4f}")
    print(f"  Recall:             {rec:.4f}")
    print(f"  Approx. Accuracy:   {((prec + rec) / 2):.4f}")
    print(f"\nTotal runtime: {time.time() - t1:.1f}s")


import os
import random
from multiprocessing import Pool, cpu_count

from tqdm import tqdm


def _worker_process_stream_sample(args):
    ex, is_math = args
    # your exact same logic:
    text = preprocess_for_fasttext(ex["text"])
    label = "MATH" if is_math else "NON_MATH"
    return f"__label__{label} {text}\n", is_math


def stream_to_files(cfg, train_path, val_path, math_iter, non_math_iter):
    total_target = cfg["training"]["total_samples"]
    math_count, non_math_count = resume_from_checkpoint(
        cfg, math_iter, non_math_iter
    )
    if math_count == total_target and non_math_count == total_target:
        print(
            f"✅ Skipping: already have {math_count} math & {non_math_count} non-math."
        )
        return

    # truncate files
    open(train_path, "w", encoding="utf-8").close()
    open(val_path, "w", encoding="utf-8").close()

    write_train = open(train_path, "a", encoding="utf-8").write
    write_val = open(val_path, "a", encoding="utf-8").write
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

    # feed into pool in moderately‐sized batches
    for result, was_math in pool.imap_unordered(
        _worker_process_stream_sample, example_generator(), chunksize=256
    ):
        # if preprocess decided to drop it, skip
        if result is None:
            continue

        # update counts
        if was_math:
            math_count += 1
        else:
            non_math_count += 1

        # write to train or val
        if random.random() < train_prob:
            write_train(result)
        else:
            write_val(result)

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


def process_file_sequentially(file_path):
    """Process a file sequentially to count math and non-math samples.

    Args:
        file_path: Path to the file to process

    Returns:
        Tuple of (math_count, non_math_count)
    """
    math_count = 0
    non_math_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__label__MATH'):
                math_count += 1
            elif line.startswith('__label__NON_MATH'):
                non_math_count += 1

    return math_count, non_math_count


def resume_from_checkpoint(cfg, math_iter, non_math_iter):
    for i in range(cfg["training"]["num_iterations"]):
        train_path = os.path.join(
            cfg["output"]["output_dir"],
            cfg["output"]["train_filename"],
            f"iteration_{i}.txt",
        )
        val_path = os.path.join(
            cfg["output"]["output_dir"],
            cfg["output"]["val_filename"],
            f"iteration_{i}.txt",
        )

        if (
            os.path.exists(train_path)
            and os.path.exists(val_path)
            and os.path.getsize(train_path) > 0
            and os.path.getsize(val_path) > 0
        ):

            # Process validation file sequentially
            math_count_val, non_math_count_val = process_file_sequentially(
                val_path
            )

            # Process train file sequentially
            math_count_train, non_math_count_train = process_file_sequentially(
                train_path
            )

            total_math = math_count_train + math_count_val
            total_non_math = non_math_count_train + non_math_count_val

            print(
                f"Iteration {i}: Found {total_math} math samples and {total_non_math} non-math samples"
            )

            # Skip iterator states with progress bars
            with tqdm(
                total=total_math, desc="Skipping math samples", unit="samples"
            ) as pbar_math:
                for _ in range(total_math):
                    next(math_iter, None)
                    pbar_math.update(1)

            with tqdm(
                total=total_non_math,
                desc="Skipping non-math samples",
                unit="samples",
            ) as pbar_non_math:
                for _ in range(total_non_math):
                    next(non_math_iter, None)
                    pbar_non_math.update(1)

            return total_math, total_non_math

    return 0, 0


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
    train_dir = os.path.join(
        cfg["output"]["output_dir"], cfg["output"]["train_filename"]
    )
    val_dir = os.path.join(
        cfg["output"]["output_dir"], cfg["output"]["val_filename"]
    )
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    math_configs = cfg["datasets"]["math_configs"]
    non_math_configs = cfg["datasets"]["non_math_configs"]

    math_ds_stream_itr_0 = load_dataset(
        cfg["datasets"]["math_dataset_name"],
        name=math_configs[0],
        split="train",
        streaming=True,
        cache_dir=cfg["datasets"]["cache_math"],
    )
    non_math_ds_stream_itr_0 = load_dataset(
        cfg["datasets"]["non_math_dataset_name"],
        name=non_math_configs[0],
        split="train",
        streaming=True,
        cache_dir=cfg["datasets"]["cache_non_math"],
    )
    math_iter = iter(math_ds_stream_itr_0)
    non_math_iter = iter(non_math_ds_stream_itr_0)

    train_path = os.path.join(
        cfg["output"]["output_dir"],
        cfg["output"]["train_filename"],
        f"iteration_{0}.txt",
    )
    val_path = os.path.join(
        cfg["output"]["output_dir"],
        cfg["output"]["val_filename"],
        f"iteration_{0}.txt",
    )

    start = time.time()
    print("→ Creating Initial Training and Validation sets")
    stream_to_files(cfg, train_path, val_path, math_iter, non_math_iter)
    print(f"✅ Created in {time.time() - start:.1f}s")

    for i in range(cfg["training"]["num_iterations"]):
        model_path = os.path.join(
            cfg["output"]["output_dir"],
            f"math_classifier_randomized_text_preprocessed_iteration_{i}.bin",
        )
        t0 = time.time()
        if not (os.path.exists(model_path) and os.path.getsize(model_path) > 0):

            train_path = os.path.join(
                cfg["output"]["output_dir"],
                cfg["output"]["train_filename"],
                f"iteration_{i}.txt",
            )
            val_path = os.path.join(
                cfg["output"]["output_dir"],
                cfg["output"]["val_filename"],
                f"iteration_{i}.txt",
            )
            print(f"Iteration - {i}. Training FastText classifier")

            model = fasttext.train_supervised(
                input=train_path, **cfg["training"]["fasttext_params"]
            )

            model.save_model(model_path)

            model.quantize()
            quant_path = model_path.replace(".bin", ".ftz")
            model.save_model(quant_path)
        else:
            model = load_model(model_path)
        t1 = time.time()
        print(
            f"✅ Trained model for iteration {i} saved to {model_path} in {t1 - t0} seconds."
        )

        print("→ Evaluating on validation set…")
        evaluate_model(model, val_path, "validation")
        print("→ Evaluating on training set…")
        evaluate_model(model, train_path, "training")
        print(f"\nEvaluation took : {time.time() - t1:.1f}s")

        if i != cfg["training"]["num_iterations"] - 1:
            t2 = time.time()
            augment_dataset(model_path, cfg, i + 1)
            print(
                f"\Data Augmentation at itr = {i} took : {time.time() - t2:.1f}s"
            )

    print(f"✅ All iterations finished in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()

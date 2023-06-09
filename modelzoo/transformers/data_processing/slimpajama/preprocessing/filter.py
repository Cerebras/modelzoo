import argparse
import os
import pickle
import queue
import re
import string
import sys
import time
from collections import defaultdict
from glob import glob
from multiprocessing import Process, Queue, cpu_count

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from lm_dataformat.lm_dataformat import Reader


def clean(s):
    # lower cased
    s = s.lower()
    # remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # remove consecutive spaces, newlines, tabs in the middle and in the beginning / end
    s = re.sub(r"\s+", " ", s.strip())
    return s


def get_short_documents(
    input_dir, threshold, n_proc, proc_idx, docs_queue, dataset_name
):
    if dataset_name == "all":
        files = []
        for dataset in [
            "arxiv",
            "stackexchange",
            "book",
            "wikipedia",
            "github",
            "c4",
            "common_crawl",
        ]:
            if dataset == "common_crawl":
                files.extend(glob(f"{input_dir}/{dataset}/*/*"))
            else:
                files.extend(glob(f"{input_dir}/{dataset}/*"))
    elif dataset_name == "common_crawl":
        files = glob(f"{input_dir}/{dataset_name}/*/*")
    else:
        files = glob(f"{input_dir}/{dataset_name}/*")
    files = sorted(files)

    for i in range(proc_idx, len(files), n_proc):
        reader = Reader(files[i])
        for doc_id, doc in enumerate(reader._stream_data(jsonl_key="text")):
            if len(clean(doc)) < threshold:
                docs_queue.put(
                    files[i].replace(input_dir + "/", "") + f"@{doc_id}"
                )


def filter_dataset(args):
    short_documents = defaultdict(set)
    docs_queue = Queue(64 * 10000)
    n_proc = cpu_count()
    processes = []
    for process_id in range(n_proc):
        p = Process(
            target=get_short_documents,
            args=(
                args.input_dir,
                args.threshold,
                n_proc,
                process_id,
                docs_queue,
                args.dataset_name,
            ),
        )
        processes.append(p)
        p.start()

    i = 0
    start_time = time.time()
    while True:
        try:
            item = docs_queue.get(timeout=30)
            file_name, doc_idx = item.split("@")
            short_documents[file_name].add(int(doc_idx))
            if i % 10 == 0:
                print(
                    f"Processed {i / args.n_docs * 100}%. ",
                    time.time() - start_time,
                )
            i += 1
        except queue.Empty:
            break

    for p in processes:
        p.join()

    print("Finished processing, writing to disk!")
    with open(args.output_file, "wb") as fout:
        pickle.dump(short_documents, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Dataset input directory.")
    parser.add_argument("output_file", help="File to output short docs to.")
    parser.add_argument(
        "n_docs", type=int, help="Number of documents in the dataset."
    )
    parser.add_argument("dataset_name")
    parser.add_argument(
        "threshold", type=int, help="Minimum length of the document to keep."
    )
    args = parser.parse_args()
    filter_dataset(args)

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

import abc
import os
import pickle
import random
from glob import glob
from multiprocessing import Manager, Process

from lm_dataformat import Reader
from tqdm import tqdm

from cerebras.modelzoo.data_preparation.nlp.slimpajama.utils import (
    cycle_documents,
    utf8len,
)


class Dataset(abc.ABC):
    def dir_path(self):
        """Path to the directory"""

    def short_documents_path(self):
        """Path to the file with short documents"""

    def name(self):
        """Human-readable name of tfhe dataset"""

    def documents(self, process_id, n_process, dup_sh, short_sh):
        """A generator producing all documents in the dataset."""
        filtered = 0
        total_count = 0
        files = glob(self.dir_path())
        random.shuffle(files)
        for file_path in files:
            reader = Reader(file_path)
            file_name = file_path.replace(self.stem_dir_path(), "")
            duplicates_set = dup_sh.get(file_name, set())
            short_set = short_sh.get(file_name, set())
            for doc_id, doc in enumerate(reader._stream_data(jsonl_key="text")):
                if doc_id % n_process == process_id:
                    if doc_id not in short_set and doc_id not in duplicates_set:
                        total_count += 1
                        yield {"doc": doc, "meta": {}}
                    else:
                        filtered += 1
        print(
            f"Total number of documents: {total_count}",
            f"Filtered documents: {filtered}",
        )

    def size(self):
        """Return an estimate of the dataset size. Implementations may use a faster, less accurate estimate."""
        size = sum(
            map(
                lambda x: utf8len(x["doc"]),
                tqdm(self.documents(), total=self.num_docs()),
            )
        )
        return size

    def num_docs(self):
        num_docs = sum(
            map(
                lambda x: 1,
                tqdm(self.documents(), total=self.num_docs()),
            )
        )
        return num_docs

    def already_shuffled(self):
        """Datasets where the source is already shuffled should override this to return True so that it isn't shuffled again."""
        return False


class RedPajamaBooksDataset(Dataset):
    def __init__(self, input_dir):
        self.stem_dir_path_ = input_dir
        self.dir_path_ = os.path.join(input_dir, "book/*.jsonl")

    def dir_path(self):
        return self.dir_path_

    def stem_dir_path(self):
        return self.stem_dir_path_

    def name(self):
        return "RedPajamaBook"

    def size(self):
        return 102851843814

    def size_duplicate_docs(self):
        return 2106014751

    def size_short_docs(self):
        return 0

    def num_docs(self):
        return 200242

    def num_duplicate_docs(self):
        return 5502

    def num_short_docs(self):
        return 0


class RedPajamaArXivDataset(Dataset):
    def __init__(self, input_dir):
        self.stem_dir_path_ = input_dir
        self.dir_path_ = os.path.join(input_dir, "arxiv/*.jsonl")

    def dir_path(self):
        return self.dir_path_

    def stem_dir_path(self):
        return self.stem_dir_path_

    def name(self):
        return "RedPajamaArXiv"

    def size(self):
        return 89018875739

    def size_duplicate_docs(self):
        return 54749418

    def size_short_docs(self):
        return 574293

    def num_docs(self):
        return 1546641

    def num_duplicate_docs(self):
        return 1979

    def num_short_docs(self):
        return 9686


class RedPajamaCommonCrawlDataset(Dataset):
    def __init__(self, input_dir):
        self.stem_dir_path_ = input_dir
        self.dir_path_ = os.path.join(input_dir, "common_crawl/*/*.jsonl.zst")

    def dir_path(self):
        return self.dir_path_

    def stem_dir_path(self):
        return self.stem_dir_path_

    def name(self):
        return "RedPajamaCommonCrawl"

    def size(self):
        return 1384835073956

    def size_duplicate_docs(self):
        return 2436638659265

    def size_short_docs(self):
        return 6867259

    def num_docs(self):
        return 187084822

    def num_duplicate_docs(self):
        return 289100390

    def num_short_docs(self):
        return 90807


class RedPajamaC4Dataset(Dataset):
    def __init__(self, input_dir):
        self.stem_dir_path_ = input_dir
        self.dir_path_ = os.path.join(input_dir, "c4/*.jsonl")

    def dir_path(self):
        return self.dir_path_

    def stem_dir_path(self):
        return self.stem_dir_path_

    def name(self):
        return "RedPajamaC4"

    def size(self):
        return 734903985384

    def size_duplicate_docs(self):
        return 53403692569

    def size_short_docs(self):
        return 664163266

    def num_docs(self):
        return 324686115

    def num_duplicate_docs(self):
        return 23015691

    def num_short_docs(self):
        return 17167086


class RedPajamaWikipediaDataset(Dataset):
    def __init__(self, input_dir):
        self.stem_dir_path_ = input_dir
        self.dir_path_ = os.path.join(input_dir, "wikipedia/*.jsonl")

    def dir_path(self):
        return self.dir_path_

    def stem_dir_path(self):
        return self.stem_dir_path_

    def name(self):
        return "RedPajamaWikipedia"

    def size(self):
        return 78649866316

    def size_duplicate_docs(self):
        return 1798885899

    def size_short_docs(self):
        return 0

    def num_docs(self):
        return 26967854

    def num_duplicate_docs(self):
        return 2866317

    def num_short_docs(self):
        return 0


class RedPajamaGithubDataset(Dataset):
    def __init__(self, input_dir):
        self.stem_dir_path_ = input_dir
        self.dir_path_ = os.path.join(input_dir, "github/*.jsonl")

    def dir_path(self):
        return self.dir_path_

    def stem_dir_path(self):
        return self.stem_dir_path_

    def name(self):
        return "RedPajamaGithub"

    def size(self):
        return 105581774510

    def size_duplicate_docs(self):
        return 90515346113

    def size_short_docs(self):
        return 0

    def num_docs(self):
        return 21232084

    def num_duplicate_docs(self):
        return 7561228

    def num_short_docs(self):
        return 0


class RedPajamaStackExchangeDataset(Dataset):
    def __init__(self, input_dir):
        self.stem_dir_path_ = input_dir
        self.dir_path_ = os.path.join(input_dir, "stackexchange/*.jsonl")

    def dir_path(self):
        return self.dir_path_

    def stem_dir_path(self):
        return self.stem_dir_path_

    def name(self):
        return "RedPajamaStackExchange"

    def size(self):
        return 71278349386

    def size_duplicate_docs(self):
        return 139373830

    def size_short_docs(self):
        return 3987870

    def num_docs(self):
        return 29702946

    def num_duplicate_docs(self):
        return 25975

    def num_short_docs(self):
        return 96165


class RedPajamaReplication(Dataset):
    def __init__(self, datasets, duplicates, short_docs):
        self.datasets = datasets
        self.duplicates = duplicates
        self.short_docs = short_docs
        self.rnd_docs = random.Random(42)
        self.rnd_queues = random.Random(420)

    def name(self):
        return "RedPajama"

    def size(self):
        return int(sum([weight * ds.size() for ds, weight in self.datasets]))

    def num_docs(self):
        """Return an estimate of the dataset number of documents.
        Implementations may use a faster, less accurate estimate."""
        return int(
            sum([ds.num_docs() * weight for ds, weight in self.datasets])
        )

    def sample_documents(
        self, weights, k, queues, process_id, n_process, dup_sh, short_sh
    ):
        # each process is going to sample documents with batch size k
        # sampling is happening globally across all available documents;
        datasets = []
        for dataset, _ in self.datasets:
            datasets.append(
                (
                    dataset.name(),
                    cycle_documents(
                        dataset, process_id, n_process, dup_sh, short_sh
                    ),
                )
            )

        for j in range(self.num_docs() // k // n_process):
            if j % 1000 == 0:
                print(f"Sampling chunk of documents {j}")
            chunk = self.rnd_docs.choices(
                population=datasets,
                weights=weights,
                k=k,
            )
            for name, documents in chunk:
                document = next(documents)
                text, meta = document["doc"], document["meta"]
                meta["redpajama_set_name"] = name
                q = self.rnd_queues.choice(queues)
                q.put({"doc": text, "meta": meta})
        print("Finished sampling documents.")

    def documents(self, queues):
        weights = []

        # calculate relative_weight for each
        total_weight = sum([x[1] * x[0].num_docs() for x in self.datasets])
        for dataset, weight in self.datasets:
            relative_weight = weight * dataset.num_docs() / total_weight
            weights.append(relative_weight)

        with open(self.duplicates, "rb") as fin:
            dup = pickle.load(fin)

        with open(self.short_docs, "rb") as fin:
            short = pickle.load(fin)

        manager = Manager()
        dup_sh = manager.dict(dup)
        short_sh = manager.dict(short)
        # create processes here to speed up read and write in shuffle_holdout.py
        # queues are given by shuffle_holdout to populate with documents
        n_process = 2 * len(queues)
        k = 1000
        procs = []
        for process_id in range(n_process):
            p = Process(
                target=self.sample_documents,
                args=(
                    weights,
                    k,
                    queues,
                    process_id,
                    n_process,
                    dup_sh,
                    short_sh,
                ),
            )
            procs.append(p)
        return procs, manager


def redpj_datasets(input_dir):
    return [
        (RedPajamaWikipediaDataset(input_dir), 1.0),
        (RedPajamaC4Dataset(input_dir), 1.0),
        (RedPajamaCommonCrawlDataset(input_dir), 1.0),
        (RedPajamaStackExchangeDataset(input_dir), 1.0),
        (RedPajamaBooksDataset(input_dir), 1.0),
        (RedPajamaGithubDataset(input_dir), 1.0),
        (RedPajamaArXivDataset(input_dir), 1.0),
    ]

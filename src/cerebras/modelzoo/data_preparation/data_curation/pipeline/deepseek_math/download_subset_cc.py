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
import gzip
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import sleep

import requests
from tqdm import tqdm

# === CONFIG ===
BASE_URL = "https://data.commoncrawl.org/"


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download a subset of Common Crawl WARC files'
    )
    parser.add_argument(
        '--warcpaths-file',
        type=str,
        required=True,
        help='Path to the gzipped WARC paths file',
    )
    parser.add_argument(
        '--download-dir',
        type=str,
        required=True,
        help='Directory to download WARC files to',
    )
    parser.add_argument(
        '--num-files',
        type=int,
        default=50,
        help='Number of WARC files to download (default: 50)',
    )
    parser.add_argument(
        '--max-threads',
        type=int,
        default=os.cpu_count() - 1,
        help='Maximum number of concurrent download threads (default: 5)',
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=2,
        help='Maximum number of retry attempts per download (default: 2)',
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1024 * 1024,
        help='Chunk size for downloading in bytes (default: 1MB)',
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)',
    )
    return parser.parse_args()


def get_random_warc_urls(path, count, random_seed):
    random.seed(random_seed)
    with gzip.open(path, "rt") as f:
        all_paths = f.read().strip().splitlines()
    sample_paths = random.sample(all_paths, count)
    return [BASE_URL + p for p in sample_paths]


def download_with_retry(url, download_dir, max_retries, chunk_size):
    filename = os.path.basename(url)
    dest_path = Path(download_dir) / filename

    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                with (
                    open(dest_path, "wb") as f,
                    tqdm(
                        desc=filename,
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        leave=False,
                    ) as bar,
                ):
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
            return True
        except Exception as e:
            print(f"[Retry {attempt}] ❌ Failed: {filename} ({e})")
            sleep(2**attempt)
    print(f"❌ Final failure: {filename}")
    return False


def main():
    args = parse_args()

    # Create download directory if it doesn't exist
    os.makedirs(args.download_dir, exist_ok=True)

    urls = get_random_warc_urls(
        args.warcpaths_file, args.num_files, args.random_seed
    )
    with ThreadPoolExecutor(max_workers=args.max_threads) as executor:
        futures = [
            executor.submit(
                download_with_retry,
                url,
                args.download_dir,
                args.max_retries,
                args.chunk_size,
            )
            for url in urls
        ]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Overall progress"
        ):
            future.result()


if __name__ == "__main__":
    main()

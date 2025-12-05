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

import requests
import wget
from bs4 import BeautifulSoup


def parse_args():
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download Common Crawl index dumps."
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default="./commoncrawlList/",
        help="Path to save the downloaded Common Crawl list.",
    )
    parser.add_argument(
        '--skip_existing_dumps',
        type=bool,
        default=True,
        help="Whether to skip dumps already downloaded to 'save_path'.",
    )
    return parser.parse_args()


def get_available_dumps(url):
    """Fetch and parse the web page to list available Common Crawl dumps."""
    response = requests.get(url)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.find_all('a', attrs={'class': "crawl-link w-inline-block"})


def main():
    args = parse_args()

    # Ensure the directory exists where the dumps will be stored.
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Get all the available dumps from Common Crawl's start page.
    url = 'https://commoncrawl.org/get-started'
    dump_links = get_available_dumps(url)

    # Prepare to track already downloaded dumps if skipping is enabled.
    existing_dumps = (
        set(os.listdir(args.save_path)) if args.skip_existing_dumps else set()
    )

    # Dumps to skip due to different file formats which are not supported for now.
    skip_list = {'CC-MAIN-2012', 'CC-MAIN-2009-2010', 'CC-MAIN-2008-2009'}

    # File to record names of newly downloaded dumps.
    with open(os.path.join(args.save_path, 'dumplist.txt'), 'w') as dump_file:
        for link in dump_links:
            dump_url = link.get('href')
            dump_name = dump_url.split('/')[-2]  # Format: 'CC-MAIN-2024-30'

            # Skip dumps either in skip list or already downloaded.
            if dump_name in skip_list or dump_name in existing_dumps:
                continue

            # Construct download URL and local save path.
            dump_list_url = dump_url.split('index.html')[0] + 'warc.paths.gz'
            dump_save_path = os.path.join(args.save_path, dump_name)

            # Ensure dump directory exists and download the dump.
            if not os.path.exists(dump_save_path):
                os.makedirs(dump_save_path)
            wget.download(dump_list_url, out=dump_save_path)
            print(f"\n Successfully downloaded {dump_name}")
            dump_file.write(dump_name + '\n')


if __name__ == '__main__':
    main()

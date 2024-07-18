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

"""
Wrapper script to download PubMed datasets
Reference: https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT
"""

import argparse
import glob
import gzip
import os
import shutil
import tarfile
import urllib.request


class Downloader:
    def __init__(self, dataset, save_path):
        """
        :param save_path: Location to download and extract the dataset
        :param dataset: One of
            "pubmed_baseline",
            "pubmed_daily_update",
            "pubmed_fulltext",
            "pubmed_open_access"

        Extracts to save_path/extracted
        """

        if dataset == "all":
            self.datasets = [
                "pubmed_baseline",
                "pubmed_daily_update",
                "pubmed_fulltext",
                "pubmed_open_access",
            ]
        else:
            self.datasets = [dataset]

        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.download_urls = {
            'pubmed_baseline': 'ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/',
            'pubmed_daily_update': 'ftp://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/',
            'pubmed_fulltext': 'ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/',
            'pubmed_open_access': 'ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/',
        }

    def download(self):
        for dataset_name in self.datasets:
            print(
                f"**** Dataset: {dataset_name}, Download_path: {self.save_path}"
            )
            url = self.download_urls[dataset_name]
            self.download_files(url, dataset_name)
            self.extract_files(dataset_name)

    def download_files(self, url, dataset):

        output = os.popen('curl ' + url).read()

        if dataset == 'pubmed_fulltext' or dataset == 'pubmed_open_access':
            line_split = (
                'comm_use' if dataset == 'pubmed_fulltext' else 'non_comm_use'
            )
            for line in output.splitlines():
                if (
                    line[-10:] == 'xml.tar.gz'
                    and line.split(' ')[-1].split('.')[0] == line_split
                ):
                    file = os.path.join(self.save_path, line.split(' ')[-1])
                    if not os.path.isfile(file):
                        print(f"Downloading: {file}")
                        response = urllib.request.urlopen(
                            url + line.split(' ')[-1]
                        )
                        with open(file, "wb") as handle:
                            shutil.copyfileobj(
                                response, handle, length=1024 * 256
                            )

        elif dataset == 'pubmed_baseline' or dataset == 'pubmed_daily_update':
            for line in output.splitlines():
                if line[-3:] == '.gz':
                    file = os.path.join(self.save_path, line.split(' ')[-1])
                    if not os.path.isfile(file):
                        print(f"Downloading {file}")
                        response = urllib.request.urlopen(
                            url + line.split(' ')[-1]
                        )
                        with open(file, "wb") as handle:
                            handle.write(response.read())

        else:
            assert False, 'Invalid PubMed dataset/dataset specified.'

    def extract_files(self, dataset):
        extractdir = os.path.join(self.save_path, 'extracted')
        if not os.path.exists(extractdir):
            os.makedirs(extractdir)

        if dataset == "pubmed_baseline" or dataset == "pubmed_daily_update":

            files = glob.glob(self.save_path + '/*.xml.gz')

            for file in files:
                print(f"Extracting: {file}")
                input = gzip.GzipFile(file, mode='rb')
                s = input.read()
                input.close()

                filename = os.path.basename(file)
                filename = filename[:-3]
                out_file = os.path.join(extractdir, filename)

                out = open(out_file, mode='wb')
                out.write(s)
                out.close()

        elif dataset == "pubmed_fulltext" or dataset == "pubmed_openaccess":
            files = glob.glob(self.save_path + '/*xml.tar.gz')

            for file in files:
                print(f"Extracting: {file}")

                filename = os.path.basename(file)
                filename = filename.split('.tar.gz')[0]
                filename = filename.replace(".", "_")

                extract_dir = os.path.join(extractdir, filename)
                with tarfile.open(file, "r:gz") as tar:
                    tar.extractall(extract_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Downloading files from PubMed'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Specify the dataset to perform --action on',
        required=True,
        choices={
            'pubmed_baseline',
            'pubmed_daily_update',
            'pubmed_fulltext',
            'pubmed_open_access',
            'all',
        },
    )
    parser.add_argument(
        '--save_path',
        type=str,
        help='Path to save the downloaded and extracted raw files',
        required=True,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    downloader = Downloader(args.dataset, args.save_path)
    downloader.download()

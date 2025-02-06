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
Script to format PubMed Fulltext commercial, PubMed Baseline and Update file Abstracts

Reference: https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT

"""

import csv
import glob
import os

import pubmed_parser as pmp


class TextFormatting:
    def __init__(
        self,
        pubmed_path,
        output_filename,
        filesize_limit=5 * (10**9),
        recursive=False,
    ):
        """
        :param str pubmed_path: Path to folder containing PubMed files
        :param str output_folder : Path to where the txt file to be written
        :param Optional[int] filesize_limit: Max size of each text file
        :param Optional[bool] recursive: Flag if true,
        searches for nxml/xml files recursively within subfolders
        """

        self.pubmed_path = pubmed_path

        print(f"self.pubmed_path:{pubmed_path}")
        self.recursive = recursive
        self.filesize = int(filesize_limit)

        self.output_folder = os.path.dirname(output_filename)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self.filename = output_filename

    def merge_abstracts(self):

        file_num = 0
        num_articles = 0
        total_articles = 0

        output_filename = (
            self.filename.split('.')[0] + f"_{int(file_num)}" + ".txt"
        )

        csv_file = output_filename.split('.')[0] + "_stats.csv"
        csv_fh = open(csv_file, 'w')
        fieldnames = ['fname', 'num_articles']
        csv_writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
        csv_writer.writeheader()

        ofile = open(output_filename, mode='w', newline='\n')
        it = glob.iglob(self.pubmed_path + '/*.xml', recursive=self.recursive)

        for filename in it:
            print(f"Processing: {filename}")

            dicts_out = pmp.parse_medline_xml(filename)
            for dict_out in dicts_out:

                if not dict_out['abstract']:
                    # Some articles have no abstract : https://pubmed.ncbi.nlm.nih.gov/13787/
                    continue
                try:
                    for line in dict_out['abstract'].splitlines():
                        if len(line) < 30:
                            # Refer to https://pubmed.ncbi.nlm.nih.gov/4969/
                            # Multiple paragraphs in abstract with subtitles such as  "Result".
                            # Removing these subtitles ONLY
                            continue
                        ofile.write(line.strip() + " ")
                    ofile.write("\n\n")

                    num_articles += 1

                except:
                    ofile.write("\n\n")
                    continue

                if int(ofile.tell()) > self.filesize:
                    ofile.close()

                    # Write to csv stats:
                    csv_writer.writerow(
                        {
                            'fname': output_filename,
                            'num_articles': num_articles,
                        }
                    )
                    total_articles += num_articles

                    # Open another file
                    file_num += 1
                    output_filename = (
                        self.filename.split('.')[0]
                        + f"_{int(file_num)}"
                        + ".txt"
                    )
                    print(f" -- Creating new file: {output_filename}")
                    ofile = open(output_filename, mode='w', newline='\n')

                    # Reset abstracts count per file
                    num_articles = 0

        total_articles += num_articles
        csv_writer.writerow(
            {'fname': output_filename, 'num_articles': num_articles}
        )
        csv_writer.writerow(
            {'fname': 'Total abstracts', 'num_articles': total_articles}
        )
        csv_fh.close()
        ofile.close()
        print(f"**** Total number of abstracts = {total_articles}")

    def merge_fulltext(self):
        # This puts one article per line

        file_num = 0
        num_articles = 0
        total_articles = 0

        output_filename = (
            self.filename.split('.')[0] + f"_{int(file_num)}" + ".txt"
        )

        csv_file = output_filename.split('.')[0] + "_stats.csv"
        csv_fh = open(csv_file, 'w')
        fieldnames = ['fname', 'num_articles']
        csv_writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
        csv_writer.writeheader()

        top_level_folders = [
            os.path.join(self.pubmed_path, x)
            for x in os.listdir(self.pubmed_path)
        ]
        top_level_folders = [x for x in top_level_folders if os.path.isdir(x)]
        print(top_level_folders)

        not_written = os.path.join(self.output_folder, "exceptions.txt")

        with open(not_written, mode='w', newline='\n') as ex_fh:

            ofile = open(output_filename, mode='w', newline='\n')

            for folder in top_level_folders:

                it = glob.iglob(folder + '/**/*.nxml', recursive=self.recursive)

                for filename in it:
                    print(f"Processing: {filename}")
                    header_dict = pmp.parse_pubmed_xml(filename)
                    body_list = pmp.parse_pubmed_paragraph(
                        filename, all_paragraph=True
                    )

                    if not header_dict and not body_list:
                        ex_fh.write(filename)
                        ex_fh.write('\n')
                        continue

                    try:
                        if header_dict:
                            ofile.write(
                                header_dict['full_title'].strip() + ". "
                            )

                        if header_dict.get('abstract', None):
                            for line in header_dict['abstract'].splitlines():
                                if len(line) < 30:
                                    continue

                                ofile.write(line.strip() + " ")

                        if body_list:
                            for dict_entry in body_list:
                                section = dict_entry['section']

                                if len(section) > 30:
                                    ofile.write(section.strip() + ". ")

                                for line in dict_entry['text'].splitlines():
                                    ofile.write(line.strip() + " ")

                            ofile.write("\n\n")

                        num_articles += 1

                    except:
                        ofile.write("\n\n")
                        continue

                    if int(ofile.tell()) > self.filesize:
                        ofile.close()

                        # Write to csv stats:
                        csv_writer.writerow(
                            {
                                'fname': output_filename,
                                'num_articles': num_articles,
                            }
                        )
                        total_articles += num_articles

                        # Open another file
                        file_num += 1
                        output_filename = (
                            self.filename.split('.')[0]
                            + f"_{int(file_num)}"
                            + ".txt"
                        )

                        print(f" -- Creating new file: {output_filename}")
                        ofile = open(output_filename, mode='w', newline='\n')

                        # Reset articles count
                        num_articles = 0

                total_articles += num_articles
                csv_writer.writerow(
                    {
                        'fname': output_filename,
                        'num_articles': num_articles,
                    }
                )
                csv_writer.writerow(
                    {
                        'fname': 'Total num articles',
                        'num_articles': total_articles,
                    }
                )
                csv_fh.close()
                ofile.close()
                print(
                    f"**** Total number of full text articles = {total_articles}"
                )

    def merge(self, dataset_name):

        if (
            dataset_name == "pubmed_baseline"
            or dataset_name == "pubmed_daily_update"
        ):
            self.merge_abstracts()

        elif (
            dataset_name == "pubmed_fulltext"
            or dataset_name == "pubmed_open_access"
        ):
            self.merge_fulltext()

        else:
            raise ValueError(f"Incorrect dataset_name: {dataset_name} passed")

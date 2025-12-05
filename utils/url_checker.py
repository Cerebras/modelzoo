"""
Cerebras docs url checker.
Used to verify all the cerebras docs links for current release.
It'll replace the published docs link with the staged URL with IP address 
and check if the URL is still valid. 
"""

import argparse
import csv
import logging
import os
import re
import requests
import subprocess

from collections import namedtuple

Url_check_info = namedtuple(
    "Url_check_info", 
    ["filename", "line_no", "orig_url", "status_code"],
)

stable_doc_domain = "https://docs.cerebras.net/en/latest"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--in_dir",
        required=True,
        help="Path to the root from which all markdown files will be scraped "
             "and checked for validity of the URLs."
    )
    parser.add_argument(
        "-r", "--root_link",
        default="",
        help="The root of the URL, i.e., collection of protocol, domain name "
             "to be replaced in the original docs link."
    )
    parser.add_argument(
        "-o", "--out_dir",
        default="./",
        help="Path to the dir at which the output csv file will be saved."
    )
    parser.add_argument(
        "--log",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="log level for logging"
    )
    parser.add_argument(
        "--check_all",
        type=bool,
        action="store_true",
        help="Do not replace with the root link and check all the links used "
             "in in_dir."
    )
    return parser.parse_args()


def get_md_files(dir_path):
    files = subprocess.check_output(["git", "ls-files"], cwd=dir_path).split()
    files = [os.path.join(dir_path, path.decode()) for path in files]
    # files = [path.decode() for path in files]
    files = [path for path in files if path.endswith(".md")]
    return set(files)


def check_url_status(url):
    try:
        get = requests.get(url)
        status_code = get.status_code
    except requests.exceptions.RequestException as e:
        logging.info(f"{url}: is not reachable \nErr: {e}")
        status_code = 0
    return status_code


def main():
    args = get_args()
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log)
    logging.basicConfig(level=numeric_level)
    md_files = get_md_files(args.in_dir)
    result = [] #collect the named tuples result here

    for file in md_files: #loop through all md files
        with open(file, "r") as f:
            lines = f.readlines()
        for line_no, line in enumerate(lines): # read lines for given file
            reo = re.search(r"https?://docs\S+.html", line)
            logging.debug(f"re match found in {file} at line no {line_no}.")
            if reo: # link found!
                # Now strip and then check the validity of the link
                link = reo.group().rstrip(".").rstrip(")")
                if not args.check_all and args.root_link != "":
                    link.replace(stable_doc_domain, args.root_link) # replace url domain
                status_code = check_url_status(link)
                if status_code != 200:
                    logging.info(
                        f"Link: {link} is not reachable, status code {status_code}."
                    )
                    link_info = Url_check_info(file, line_no, link, status_code)
                    result.append(link_info)
    
    with open(os.path.join(args.out_dir, "links_check_result.csv"),'w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(Url_check_info._fields)
        for row in result:
            csv_out.writerow(row)


if __name__ == "__main__":
    main()

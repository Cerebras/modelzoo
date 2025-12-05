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

import random

from lxml import html
from prettytable import PrettyTable


def html_table_to_ascii(headers, rows):
    ascii_table = PrettyTable()
    if headers:
        ascii_table.field_names = headers
    else:
        raise ValueError("Headers cannot be empty when using PrettyTable.")

    for row in rows:
        # check if row is empty, and ensure the number of columns is consistent
        if row and len(row) != len(headers):
            raise ValueError(
                f"Row has incorrect number of values: {len(row)} != {len(headers)}"
            )
        ascii_table.add_row(row)

    return ascii_table.get_string()


def html_table_to_markdown(headers, rows):
    markdown_table = []
    if headers:
        markdown_table.append("| " + " | ".join(headers) + " |")
        markdown_table.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        if row and any(cell.strip() for cell in row):  # 确保行不是空的
            formatted_row = [cell.replace("\n", " ").strip() for cell in row]
            markdown_table.append("| " + " | ".join(formatted_row) + " |")
    return "\n".join(markdown_table)


def random_table_converter(table_element, format_choice=None):
    format_choice = format_choice or random.choice(["ascii", "markdown"])
    headers = [
        th.text_content().strip() for th in table_element.xpath(".//th")
    ] or []
    rows = [
        [td.text_content().strip() for td in tr.xpath(".//td")]
        for tr in table_element.xpath(".//tr")
        if tr.xpath(".//td")
    ]
    if format_choice == "ascii":
        return html_table_to_ascii(headers, rows)
    else:
        return html_table_to_markdown(headers, rows)


def process_tables(tree, format_choice=None):
    if not isinstance(tree, html.HtmlElement):
        raise TypeError("Expected an lxml.html.HtmlElement object")
    for table in tree.xpath("//table"):
        table_text = random_table_converter(table, format_choice=format_choice)
        new_element = html.Element("pre", attrib={"class": "converted-table"})
        new_element.text = f"\n{table_text}\n"
        table.getparent().replace(table, new_element)
    return tree

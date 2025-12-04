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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys

from data_curation.pipeline.deepseek_math.mathml2latex.unicode_map import (
    unicode_map,
)
from lxml import etree

# MathML to LaTeX conversion with XSLT from Vasil Yaroshevich
base_path = os.path.dirname(os.path.realpath(__file__))
xslt_file = os.path.join(base_path, 'mmltex', 'mmltex.xsl')
xslt = etree.parse(xslt_file)
transform = etree.XSLT(xslt)


# add by zzwang


def preprocess_and_parse_xml(xml_content):
    # 替换常见的 HTML 实体
    # entity_replacements = {
    #     '&nbsp;': '&#160;',  # 非断空格
    #     '&lt;': '&lt;',      # 小于号
    #     '&gt;': '&gt;',      # 大于号
    #     '&amp;': '&amp;',    # &符号
    #     '&quot;': '&quot;',  # 双引号
    #     '&apos;': '&apos;',  # 单引号
    # }

    # for entity, replacement in entity_replacements.items():
    #     xml_content = xml_content.replace(entity, replacement)

    # # 移除或替换其他可能导致问题的字符
    # xml_content = re.sub(r'&#x([0-9a-fA-F]+);', lambda m: chr(int(m.group(1), 16)), xml_content)
    # xml_content = re.sub(r'&#([0-9]+);', lambda m: chr(int(m.group(1))), xml_content)

    # 尝试解析预处理后的内容
    try:
        return etree.fromstring(xml_content)
    except etree.XMLSyntaxError as e:
        print(f"解析错误: {e}")
        # 如果仍然失败，可以尝试使用更宽松的解析器
        parser = etree.XMLParser(recover=True)
        return etree.fromstring(xml_content, parser)


def mathml2latex(mathml_block):
    # Preprocess to remove aliases
    mathml_block = mathml_block.replace('<<', '&lt;<').replace('>>', '>&gt;')
    # dom = etree.fromstring(mathml_block)
    dom = preprocess_and_parse_xml(mathml_block)
    return transform(dom)


def unicode2latex(latex_block):
    latex_text = str(latex_block, 'utf-8').encode('ascii', 'backslashreplace')
    for utf_code, latex_code in unicode_map.items():
        latex_text = str(latex_text).replace(utf_code, latex_code)
    latex_text = latex_text.replace('\\\\', '\\')  # "\\" --> "\"
    latex_text = re.sub(
        r'\\textcolor\[rgb\]\{[0-9.,]+\}', '', latex_text
    )  # "\textcolor[rgb]{...}" --> ""
    latex_text = latex_text.replace('\\ ~\\ ', '{\\sim}')  # " ~ " --> "{\sim}"
    latex_text = latex_text[len('b\'') :][: -len('\'')]  # b'...' --> ...
    latex_text = re.sub(r'^\$ ', '$', latex_text)  # "$ " --> "$"
    latex_text = latex_text.replace('{\\ }', '\\ ')  # "{ }" --> " "
    latex_text = re.sub(r' \}', '}', latex_text)  # " }" --> "}"
    latex_text = latex_text.replace('\\n\\[\\n\\t', '$$').replace(
        '\\n\\]', '$$'
    )
    return latex_text


def convert(text):
    mathml_blocks = re.findall(r"<!--\[if mathML\]>(.*?)<!\[endif\]-->", text)
    for mathml_block in mathml_blocks:
        latex_block = mathml2latex(mathml_block)
        latex_text = unicode2latex(latex_block)
        text = text.replace(
            '<!--[if mathML]>' + mathml_block + '<![endif]-->', latex_text
        )
    # Remove multiple consecutive blank lines
    for _ in range(2):
        text = re.sub(r'\n\n', '\n', text)
    return text


def main():
    input_file = open(sys.argv[1], "r", encoding="utf-8")
    input = input_file.read()
    input_file.close()
    output = convert(input)
    output_file = open(sys.argv[2], "w", encoding="utf-8")
    output_file.write(output)
    output_file.close()


# if __name__ == "__main__":
#     main()

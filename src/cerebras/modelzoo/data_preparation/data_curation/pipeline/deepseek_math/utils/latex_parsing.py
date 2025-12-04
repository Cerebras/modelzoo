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

import re
import sys
import time

from data_curation.pipeline.deepseek_math.mathml2latex.mathml2latex import (
    mathml2latex,
    unicode2latex,
)
from resiliparse.parse.html import HTMLTree, traverse_dom
from resiliparse.process_guard import (
    ExecutionTimeout,
    InterruptType,
    MemoryLimitExceeded,
    mem_guard,
    time_guard,
)


def improve_latex_content_parsing(html_doc):
    tree = HTMLTree.parse(html_doc)

    def remove_math_styles(latex_text):
        if "display" not in latex_text and "textstyle" not in latex_text:
            return latex_text

        pattern = r"\$\{?(\\(?:display|text)style)\s*(.+?)\}?\$"

        def replace_func(match):
            content = match.group(2)
            content = re.sub(r"^\{(.+)\}$", r"\1", content)
            return f"${content}$"

        cleaned_text = re.sub(pattern, replace_func, latex_text)
        return cleaned_text

    def clean_latex(latex_text):
        latex_text = latex_text.strip()
        if latex_text.startswith("{\\displaystyle"):
            latex_text = latex_text.replace("{\\displaystyle", "")
            if latex_text.endswith("}"):
                latex_text = latex_text[:-1]
        if latex_text.strip() == "":
            return ""
        return f"${latex_text.strip()}$"

    def process_math_element(math_elem):
        if math_elem.getattr("class") == "katex-mathml":
            print("skip katex mathml")
            return  # 跳过 KaTeX 的 HTML/CSS 渲染部分

        latex = extract_latex_with_timeout(math_elem)
        if latex == None:
            return

        new_span = tree.create_element("span")
        new_span["class"] = "math-text"
        new_span.text = latex.strip()
        parent = math_elem.parent
        if parent:
            parent.replace_child(new_span, math_elem)

    def clean_mathml(mathml_block):
        if "oldsymbol{" in mathml_block and "boldsymbol{" not in mathml_block:
            mathml_block = mathml_block.replace("oldsymbol", "\\boldsymbol")
        mathml_block = re.sub(r"<\?xml[^>]+\?>\s*", "", mathml_block)
        if 'xmlns="http://www.w3.org/1998/Math/MathML"' not in mathml_block:
            mathml_block = mathml_block.replace(
                "<math", '<math xmlns="http://www.w3.org/1998/Math/MathML"', 1
            )
        return mathml_block

    def extract_latex(elem):
        annotation = elem.query_selector(
            'annotation[encoding="application/x-tex"]'
        )
        if annotation and annotation.text:
            return clean_latex(annotation.text)

        if "alttext" in elem:
            return clean_latex(elem["alttext"])

        elem = clean_mathml(str(elem))
        latex_block = mathml2latex(elem)
        latex_text = unicode2latex(latex_block)
        latex_text = remove_math_styles(latex_text)
        return latex_text

    def extract_latex_with_timeout(elem, timeout=0.1):
        import threading

        result = [None]
        exception = [None]
        start_time = time.time()

        def target():
            try:
                result[0] = extract_latex(elem)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            return None  # or return the original math_elem, or a simplified version

        if exception[0] is not None:
            raise exception[0]

        elapsed_time = time.time() - start_time
        return result[0]

    def parse_katex_html(element):
        """递归解析 KaTeX HTML 结构，精准提取上下标"""
        text = ""
        for child in element.child_nodes:
            if child.tag != "span":
                continue

            classes = child.class_name.split() if child.class_name else []
            if "mord" in classes:  # mo character (e.g. F)
                text += parse_katex_html(child) or child.text
            elif "msupsub" in classes:  # sup/sub container
                subscript = ""
                # locate the vlist structure containing the subscript content
                vlist_t = child.query_selector(".vlist-t")
                if vlist_t:
                    for vlist_r in vlist_t.child_nodes:
                        if (
                            "vlist-r" in vlist_r.class_name.split()
                            if vlist_r.class_name
                            else []
                        ):
                            for vlist in vlist_r.child_nodes:
                                if (
                                    "vlist" in vlist.class_name.split()
                                    if vlist.class_name
                                    else []
                                ):
                                    # extract the content in all sizing layers
                                    sizing = vlist.query_selector(".sizing")
                                    if sizing:
                                        subscript += parse_katex_html(sizing)
                # filter the zero-width space and concatenate the subscript
                subscript = subscript.replace(
                    "​", ""
                )  # remove the Unicode zero-width space
                if subscript:
                    text += f"_{{{subscript}}}"
            else:
                # recursively process other nested structures
                text += parse_katex_html(child)

        # prioritize the content of the child node, then the text of the self
        return text.strip() or (element.text.strip() if element.text else "")

    def process_katex_container(katex_elem):
        mathml_elem = katex_elem.query_selector(".katex-mathml math")
        if mathml_elem:
            parent = katex_elem.parent
            if parent:
                print(">>>>>>>processing by katex mathml")
                parent.replace_child(mathml_elem, katex_elem)
                return

        for html_katex in katex_elem.get_elements_by_class_name("katex-html"):
            parent = katex_elem.parent
            if not parent:
                continue

            math_text = parse_katex_html(html_katex)

            math_text = math_text.replace("\u200b", " ")

            if math_text.strip() == "":
                continue
            print(">>>>>>>processing by katex html")

            new_span = tree.create_element("span")
            new_span["class"] = "math-text"
            new_span.text = f"${math_text}$"
            parent.replace_child(new_span, katex_elem)

    def process_math_html_entities(tree):
        # replace the math symbols
        # replacements = {
        #     "&minus;": "-",
        #     "&radic;": "\\sqrt",
        #     "&gamma;": "\\gamma",
        # }
        def start_callback(context):
            node = context.node

            def replace_sub_sup_tag(node):
                if node.tag == "sup" and not node.text.startswith("^{"):
                    node.text = f"^{{{node.text}}}"
                elif node.tag == "sub" and not node.text.startswith("_{"):
                    node.text = f"_{{{node.text}}}"

                for child in node.child_nodes:
                    replace_sub_sup_tag(
                        child
                    )  # directly process the child nodes

                return node

            node = replace_sub_sup_tag(node)

            # replace the <span class="intbl"> tag
            if node.tag == "span" and "intbl" in node.getattr("class", ""):
                numerator = node.get_elements_by_tag_name("em")

                denominator = node.get_elements_by_tag_name("strong")
                if numerator and denominator:
                    # print("=="*10)
                    node.text = f"\\frac{{{numerator[0].text}}}{{{denominator[0].text}}}"
                    # print(node.text)

        # traverse the DOM tree from the body
        traverse_dom(
            tree.body,
            start_callback=start_callback,
            elements_only=False,  # traverse all nodes, including text nodes
        )

        # return the modified DOM tree
        return tree

    if tree.body is not None:
        # process the KaTeX container
        for katex_elem in tree.body.get_elements_by_class_name("katex"):
            process_katex_container(katex_elem)

        for math_elem in tree.body.get_elements_by_tag_name("math"):
            process_math_element(math_elem)

        # for key, value in MATH_HTML_ENTITIES_REPLACEMENTS.items():
        #     if key in tree.body.html:
        #         print("=="*10)
        #         print(f"{key} -> {value}")
        #         tree.body.html = tree.body.html.replace(key, value)
        tree = process_math_html_entities(tree)
    else:
        print("Warning: The HTML document has no body.")

    return str(tree)


def improve_latex_content_parsing_with_timeout(html_doc):
    with mem_guard(
        max_memory=1024 * 1024 * 4, interrupt_type=InterruptType.exception
    ):  # 4GB limit
        with time_guard(
            timeout=0.1, interrupt_type=InterruptType.exception
        ) as guard:  # 1 second timeout
            try:
                return improve_latex_content_parsing(html_doc)
            except ExecutionTimeout:
                sys.stderr.write("Timeout! Returning original HTML.\n")
                sys.stderr.flush()
                return html_doc
            except MemoryLimitExceeded:
                sys.stderr.write(
                    "Memory limit exceeded! Returning original HTML.\n"
                )
                sys.stderr.flush()
                return html_doc

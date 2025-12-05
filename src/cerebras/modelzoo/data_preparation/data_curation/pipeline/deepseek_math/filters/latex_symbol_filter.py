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

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter


class LatexSymbolFilter(BaseFilter):
    """
    Filter documents by eng content classification using FastText.
    """

    name = "LatexSymbolFilter"

    MATH_KEYWORDS = [
        'MathJax',
        'mathjax',
        '<math',
        'math-container',
        'katex.min.css',
        'latex.php',
        'codecogs',
        'tex.cgi',
        'class="tex"',
        "class='tex'",
    ]

    latex_math_commands = [
        "\\end",
        "\\begin",
        "\\ref",
        "\\frac",
        "\\label",
        "\\bf",
        "\\right",
        "\\left",
        "\\rm",
        "\\alpha",
        "\\mu",
        "\\def",
        "\\it",
        "\\pi",
        "\\sigma",
        "\\sum",
        "\\lambda",
        "\\beta",
        "\\nu",
        "\\partial",
        "\\int",
        "\\delta",
        "\\rho",
        "\\phi",
        "\\gamma",
        "\\omega",
        "\\over",
        "\\nonumber",
        "\\bar",
        "\\sqrt",
        "\\theta",
        "\\tau",
        "\\em",
        "\\rangle",
        "\\hat",
        "\\tilde",
        "\\cal",
        "\\hline",
        "\\item",
        "\\psi",
        "\\vec",
        "\\langle",
        "\\epsilon",
        "\\eta",
        "\\cdot",
        "\\in",
        "\\xi",
        "\\infty",
        "\\quad",
        "\\mathcal",
        "\\times",
        "\\emph",
        "\\mathbf",
        "\\prime",
        "\\be",
        "\\mathrm",
        "\\ee",
        "\\vspace",
        "\\pm",
        "\\chi",
        "\\ell",
        "\\text",
        "\\qquad",
        "\\noindent",
        "\\to",
        "\\varphi",
        "\\hspace",
        "\\leq",
        "\\cos",
        "\\eqref",
        "\\overline",
        "\\sin",
        "\\kappa",
        "\\hbox",
        "\\rightarrow",
        "\\varepsilon",
        "\\textit",
        "\\dagger",
        "\\big",
        "\\otimes",
        "\\equiv",
        "\\zeta",
        "\\dot",
        "\\ln",
    ]
    latex_regex = re.compile('\\\\[a-z]{2,}')
    original_regex = re.compile('|'.join(MATH_KEYWORDS))

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def filter(self, doc: Document) -> bool:
        ## Currently this filter returns false when latex symbol is present else returns true.
        ## Sets metadata - "contains_latex_symbols" to true or false based on a regex search.

        original_match = LatexSymbolFilter.original_regex.search(doc.text)
        if original_match:
            doc.metadata["contains_latex_symbols"] = 1
            return True
        else:
            latex_match = LatexSymbolFilter.latex_regex.search(doc.text)
            if latex_match:
                for term in LatexSymbolFilter.latex_math_commands:
                    if term in doc.text:
                        doc.metadata["contains_latex_symbols"] = 1
                        return True
        doc.metadata["contains_latex_symbols"] = 0
        return True

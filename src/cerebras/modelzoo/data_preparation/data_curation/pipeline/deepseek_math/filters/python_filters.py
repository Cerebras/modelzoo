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

import contextlib
import os
import re
import sys
import tempfile
import tokenize
from io import StringIO
from typing import List, Optional, Tuple

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.utils.logging import logger
from pylint.lint import Run


class PythonSyntaxFilter(BaseFilter):
    """Filter that checks Python code for syntax errors."""

    name = "🐍 Python Syntax"

    def __init__(self, **kwargs):  # Accept kwargs
        super().__init__(**kwargs)  # Pass kwargs to parent

    def filter(self, doc: Document) -> bool:
        """Returns True if the document passes the filter (no syntax errors)."""
        if not doc.metadata.get("language") == "Python":
            return True  # Pass non-Python documents

        code = doc.text
        try:
            compile(code, "<string>", "exec")
            return True
        except SyntaxError as e:
            doc.metadata["syntax_error"] = str(e)
            logger.info(
                f"Document {doc.metadata.get('filepath', 'unknown')} failed syntax check."
            )
            return False
        except MemoryError as e:
            doc.metadata["memory_error"] = str(e)
            logger.info(
                f"Document {doc.metadata.get('filepath', 'unknown')} got memory error."
            )
            return False

        except RecursionError as e:
            # Handle recursion errors gracefully
            doc.metadata["recursion_error"] = str(e)
            logger.info(
                f"Document {doc.metadata.get('filepath', 'unknown')} got recursion error."
            )
            return False


class PythonPylintScoreFilter(BaseFilter):
    """Filter based on Pylint code quality score.
    Apply penalty for high comment ratio if configured."""

    name = "✨ Python Pylint Score"

    def __init__(
        self,
        min_score: float = 0.0,
        apply_comment_penalty: bool = True,
        **kwargs,
    ):
        super().__init__(
            **kwargs
        )  # Pass kwargs (including exclusion_writer) to parent
        self.min_score = min_score
        self.apply_comment_penalty = apply_comment_penalty

        # Increase recursion limit for Pylint
        sys.setrecursionlimit(5000)  # Increase from default 1000

        self.pylint_args = [
            "--persistent=n",
            "--disable=all",
            "--enable=F,E,W,R,C",
            "--disable=E0401,C0114,C0301,C0103,C0116,C0411,R0903,W0511,C0412",
            "--score=y",
            "--reports=n",
            "--msg-template=''",
        ]

    def get_pylint_score(self, code: str) -> Optional[float]:
        """Use pylint API directly - much faster than subprocess."""
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".py", mode='w'
        ) as f:
            f.write(code)
            f.flush()

            try:
                # Suppress output
                with (
                    contextlib.redirect_stdout(StringIO()),
                    contextlib.redirect_stderr(StringIO()),
                ):
                    results = Run([f.name] + self.pylint_args, exit=False)

                score = results.linter.stats.global_note
                return float(score) if score is not None else None
            except:
                return None
            finally:
                try:
                    os.unlink(f.name)
                except:
                    pass

    def apply_comment_penalty_to_score(
        self, score: float, comment_ratio: float
    ) -> float:
        """Apply penalty to score based on comment ratio."""
        if comment_ratio == 1.0:
            return 0.0
        elif comment_ratio > 0:
            penalty_factor = 1 - comment_ratio
            score *= penalty_factor
        return score

    def calculate_comment_ratio(self, code: str) -> float:
        """Calculate the ratio of comment lines to total lines."""
        total_lines = 0
        comment_lines = 0

        try:
            tokens = tokenize.generate_tokens(StringIO(code).readline)
            for token_type, _, _, _, _ in tokens:
                total_lines += 1
                if token_type == tokenize.COMMENT:
                    comment_lines += 1
        except (tokenize.TokenError, IndentationError):
            return 0

        if total_lines == 0:
            return 0

        return comment_lines / total_lines

    def filter(self, doc: Document) -> bool:
        """Returns True if Pylint score meets minimum threshold."""
        if not doc.metadata.get("language") == "Python":
            return True

        try:
            score = self.get_pylint_score(doc.text)
            if score is None:
                return False

            if self.apply_comment_penalty:
                comment_ratio = self.calculate_comment_ratio(doc.text)
                score = self.apply_comment_penalty_to_score(
                    score, comment_ratio
                )

            doc.metadata["pylint_score"] = score
            return score >= self.min_score
        except RecursionError as e:
            # Skip files that cause recursion errors
            logger.info(
                f"RecursionError for document: {doc.metadata.get('filepath', 'unknown')}"
            )
            doc.metadata["pylint_error"] = str(e)
            return False  # Filter out problematic files
        except MemoryError as e:
            # Skip files that cause memory errors
            logger.info(
                f"MemoryError for document: {doc.metadata.get('filepath', 'unknown')}"
            )
            doc.metadata["pylint_error"] = str(e)
            return False


class PythonLanguageCharacterFilter(BaseFilter):
    """Filter that checks for non-English/Japanese characters in Python code."""

    name = "🌐 Python Language Characters"

    def __init__(
        self,
        allowed_pattern: str = r"^[\u0020-\u007E\u3000-\u30FF\u4E00-\u9FFF\uFF66-\uFF9F\s\n\t]*$",
        **kwargs,
    ):
        super().__init__(**kwargs)  # Pass kwargs to parent
        self.allowed_characters = re.compile(allowed_pattern)

    def check_language_issues(self, code: str) -> Tuple[List[str], str]:
        """Check for language/character issues in the code."""
        issues = []
        language_type = "English or Japanese"

        try:
            tokens = tokenize.generate_tokens(StringIO(code).readline)
            for token_type, token_string, _, _, _ in tokens:
                if token_type in {tokenize.STRING, tokenize.COMMENT}:
                    if not self.allowed_characters.match(token_string):
                        truncated = (
                            token_string[:100] + "..."
                            if len(token_string) > 100
                            else token_string
                        )
                        issues.append(
                            f"Non-Japanese/English characters found in: {truncated}"
                        )
                        language_type = (
                            "Contains non-English/Japanese characters"
                        )
                        break
        except tokenize.TokenError as e:
            issues.append(f"Token error: {str(e)}")
            language_type = "TokenError in code"
        except IndentationError as e:
            issues.append(f"Indentation error: {str(e)}")
            language_type = "IndentationError in code"

        return issues, language_type

    def filter(self, doc: Document) -> bool:
        """Returns True if no language character issues are found."""
        if not doc.metadata.get("language") == "Python":
            return True

        issues, language_type = self.check_language_issues(doc.text)

        doc.metadata["language_issues"] = issues
        doc.metadata["language_type"] = language_type

        if len(issues) > 0:
            logger.info(
                f"Document {doc.metadata.get('filepath', 'unknown')} failed language character check: {language_type}"
            )
            doc.metadata["language_character_error"] = ','.join(issues)
            return False
        else:
            return True

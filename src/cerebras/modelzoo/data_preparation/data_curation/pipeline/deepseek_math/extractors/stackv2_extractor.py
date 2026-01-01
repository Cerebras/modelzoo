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

#!/usr/bin/env python3
"""
PyPI project extraction pipeline step for datatrove.

Input Document.text: JSON like
{
  "id": "...",
  "file_contents": { "/path.py": "<code>", ... },
  "metadata": { ... }
}

Output Document.text: JSON like
{
  "id": "...",
  "python_code": { "a.py": "...", ... },            # topo-ordered (non-tests)
  "python_tests_code": { "tests/test_x.py": "..."}, # topo-ordered among tests
  # optional when present:
  # "cpp_code": {...}, "rust_code": {...}, ...
  "metadata": { ... },                               # copied from input
  "dir_structure": "root\n|- ...",
}
"""
import ast
import json
from collections import defaultdict, deque
from typing import Dict, Generator, Iterable, List, Set, Tuple

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import logger

# -----------------------------
# Language extension map
# -----------------------------
LANGUAGE_EXTENSIONS = {
    "python": {".py", ".pyi"},
    "cpp": {".cpp", ".h", ".hpp", ".cc", ".cxx", ".c++"},
    "rust": {".rs"},
    "java": {".java"},
    "c": {".c", ".h"},
    "javascript": {".js"},
    "typescript": {".ts"},
    "go": {".go"},
    "ruby": {".rb"},
    "php": {".php"},
    "swift": {".swift"},
}
PY_EXTS = LANGUAGE_EXTENSIONS["python"]

# Prefer these module roots if present in paths
CANDIDATE_MODULE_ROOTS = (
    "src/",
    "",
)  # string prefixes against normalized relpaths

TEST_PATTERNS = (
    "/tests/",
    "/test/",
)


def _is_test(rel: str, base_name: str) -> bool:
    return (
        any(p in rel for p in TEST_PATTERNS)
        or base_name.startswith("test_")
        or base_name.endswith("_test.py")
    )


def _split_ext(path: str) -> Tuple[str, str]:
    idx = path.rfind(".")
    return (path, "") if idx < 0 else (path[:idx], path[idx:])


def _norm_rel(path: str) -> str:
    # normalize to POSIX-ish relative path without leading slash
    return path.lstrip("/")


def _build_tree_string(paths: List[str]) -> str:
    # paths are normalized relative paths ('a/b/c.py')
    # build a simple tree string
    root = {}
    for p in paths:
        parts = p.split("/")
        node = root
        for i, seg in enumerate(parts):
            node = node.setdefault(seg, {} if i < len(parts) - 1 else None)

    lines = ["root"]

    def rec(node, prefix=""):
        for name in sorted(node.keys()):
            lines.append(f"{prefix}|- {name}")
            child = node[name]
            if isinstance(child, dict):
                rec(child, prefix + "|  ")

    rec(root)
    return "\n".join(lines)


# ---------- Import parsing from source ----------
def _parse_imports_from_source(src: str, cur_mod: str) -> Set[str]:
    """
    Returns absolute-ish imported module names (best-effort) from Python source.
    Handles:
      - import a.b.c
      - from x.y import z
      - from ..x import y   (relative; resolved against cur_mod)
    """
    out: Set[str] = set()
    try:
        node = ast.parse(src)
    except Exception:
        return out
    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name:
                    out.add(a.name)
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ""
            if n.level and cur_mod:
                # resolve "from ..base import ..." relative to cur_mod
                parts = cur_mod.split(".")
                up = max(0, len(parts) - n.level)
                prefix = ".".join(parts[:up])
                base = f"{prefix}.{base}" if base else prefix
                base = base.strip(".")
            if base:
                out.add(base)
    return out


# ---------- Module name helpers ----------
def _compute_module_name(rel: str, module_roots: List[str]) -> str:
    """
    Turn 'pkg/sub/mod.py' into 'pkg.sub.mod'; '__init__.py' maps to package module.
    Apply the first matching module root (e.g., 'src/').
    """
    path = rel
    for root in module_roots:
        if root and path.startswith(root):
            path = path[len(root) :]
            break
    # strip .py/.pyi
    if path.endswith("/__init__.py"):
        mod = path[: -len("/__init__.py")].replace("/", ".")
        return mod
    for ext in (".py", ".pyi"):
        if path.endswith(ext):
            return path[: -len(ext)].replace("/", ".")
    return path.replace("/", ".")


def _choose_module_roots(all_rels: List[str]) -> List[str]:
    # If any file starts with 'src/', prefer that root, else '' (project root)
    has_src = any(r.startswith("src/") for r in all_rels)
    return ["src/"] if has_src else [""]


# ---------- Graph building & topo ----------
def _build_dep_graph_from_memory(
    code_rels: List[str],
    rel_to_src: Dict[str, str],
    module_roots: List[str],
) -> Dict[str, Set[str]]:
    """
    Build A->deps graph among 'code_rels' using in-memory sources.
    """
    file_to_mod: Dict[str, str] = {
        rel: _compute_module_name(rel, module_roots) for rel in code_rels
    }
    # provides: module -> files
    provides: Dict[str, Set[str]] = defaultdict(set)
    for rel, mod in file_to_mod.items():
        provides[mod].add(rel)

    deps: Dict[str, Set[str]] = {rel: set() for rel in code_rels}
    for rel in code_rels:
        src = rel_to_src.get(rel, "")
        cur_mod = file_to_mod.get(rel, "")
        imported = _parse_imports_from_source(src, cur_mod)
        for name in imported:
            targets = provides.get(name)
            if not targets:
                # try longest prefix match: pkg.sub.mod -> pkg.sub -> pkg
                parts = name.split(".")
                while parts:
                    prefix = ".".join(parts)
                    if prefix in provides:
                        targets = provides[prefix]
                        break
                    parts.pop()
            if targets:
                deps[rel].update(targets)
        deps[rel].discard(rel)
    return deps


def _topo_order_break_cycles(deps: Dict[str, Set[str]]) -> List[str]:
    """
    Kahn + tie-breaker: pick node with FEWEST deps, then lexicographic.
    """
    reverse: Dict[str, Set[str]] = defaultdict(set)
    indeg: Dict[str, int] = {}
    for u, vs in deps.items():
        indeg[u] = len(vs)
        for v in vs:
            reverse[v].add(u)

    zeros = deque(sorted([u for u, d in indeg.items() if d == 0]))
    remaining = set(deps.keys())
    out: List[str] = []

    while remaining:
        if zeros:
            u = zeros.popleft()
        else:
            u = min(remaining, key=lambda x: (len(deps[x]), x))
        out.append(u)
        remaining.remove(u)
        for w in list(reverse.get(u, ())):
            if w in remaining:
                if u in deps[w]:
                    deps[w].remove(u)
                indeg[w] = max(0, indeg[w] - 1)
                if indeg[w] == 0:
                    # keep sorted for determinism
                    inserted = False
                    for i, cur in enumerate(zeros):
                        if w < cur:
                            zeros.insert(i, w)
                            inserted = True
                            break
                    if not inserted:
                        zeros.append(w)
    return out


# -----------------------------
# The pipeline step
# -----------------------------
class StackV2Extractor(PipelineStep):
    name = "🐍 PyPI Extractor"
    type = "🔧 processor"

    def run(
        self, data: Iterable[Document], rank: int = 0, world_size: int = 1
    ) -> Generator[Document, None, None]:
        for doc in data:
            try:
                # Parse input JSON from doc.text (source-of-truth)

                repo_id: str = doc.text.get("repo_name") or doc.id or ""
                file_contents: Dict[str, str] = doc.text
                in_metadata: Dict[str, object] = doc.metadata

                # Normalize relpaths and bucket by language
                rel_to_src: Dict[str, str] = {}
                lang_buckets: Dict[str, Dict[str, str]] = defaultdict(dict)

                all_rels: List[str] = []
                for raw_rel, content in file_contents.items():
                    rel = _norm_rel(raw_rel)
                    all_rels.append(rel)
                    rel_to_src[rel] = content
                    _, ext = _split_ext(rel)
                    ext = ext.lower()
                    # language assignment
                    assigned = False
                    for lang, exts in LANGUAGE_EXTENSIONS.items():
                        if ext in exts:
                            lang_buckets[lang][rel] = content
                            assigned = True
                            break
                    if not assigned:
                        # ignore non-code for this step (spec doesn't require non-code output)
                        pass

                # Dir structure from all paths
                dir_structure = _build_tree_string(all_rels)

                # Split python into code vs tests
                py_files = list(lang_buckets.get("python", {}).keys())
                py_tests = [
                    r for r in py_files if _is_test(r, r.rsplit("/", 1)[-1])
                ]
                py_code = [r for r in py_files if r not in py_tests]

                # Choose module roots (prefer src/ layout)
                module_roots = _choose_module_roots(py_files)

                # Topo for python code
                code_deps = _build_dep_graph_from_memory(
                    py_code, rel_to_src, module_roots
                )
                code_order = _topo_order_break_cycles(code_deps)

                # Topo for python tests (consider only tests graph)
                tests_deps = _build_dep_graph_from_memory(
                    py_tests, rel_to_src, module_roots
                )
                tests_order = _topo_order_break_cycles(tests_deps)

                # Build ordered dicts (insertion order preserved in Python 3.7+)
                python_code_out: Dict[str, str] = {
                    rel: rel_to_src[rel] for rel in code_order
                }
                python_tests_out: Dict[str, str] = {
                    rel: rel_to_src[rel] for rel in tests_order
                }
                py_src_code_ast = {}
                for filename, code in python_code_out.items():
                    try:
                        py_code_ast = ast.parse(code)
                        ast_string = ast.dump(py_code_ast, indent=2)
                        py_src_code_ast[filename] = ast_string
                        self.stat_update("python_src_ast_extraction_passed")
                    except SyntaxError as e:
                        logger.error(
                            f"Unable to get ast of file - {filename}. Error - {e}"
                        )
                        self.stat_update("python_src_ast_extraction_failed")
                py_test_code_ast = {}
                for filename, code in python_tests_out.items():
                    try:
                        py_code_ast = ast.parse(code)
                        ast_string = ast.dump(py_code_ast, indent=2)
                        py_test_code_ast[filename] = ast_string
                        self.stat_update("python_test_ast_extraction_passed")
                    except SyntaxError as e:
                        logger.error(
                            f"Unable to get ast of file - {filename}. Error - {e}"
                        )
                        self.stat_update("python_test_ast_extraction_failed")

                # Other languages (no topo requested) – include only if present
                extra_lang_maps: Dict[str, Dict[str, str]] = {}
                for lang, mapping in lang_buckets.items():
                    if lang == "python":
                        continue
                    if mapping:
                        key = f"{lang}_code"
                        extra_lang_maps[key] = {
                            rel: mapping[rel] for rel in sorted(mapping.keys())
                        }

                out_row = {
                    "id": repo_id,
                    "python_code": python_code_out,
                    "python_tests_code": python_tests_out,
                    "python_src_ast": py_src_code_ast,
                    "python_test_ast": py_test_code_ast,
                    **extra_lang_maps,
                    "metadata": in_metadata,  # unchanged
                    "dir_structure": dir_structure,
                }

                yield Document(
                    text=json.dumps(out_row, ensure_ascii=False),
                    id=repo_id,
                    metadata={},
                )
                self.stat_update("processed_ok")

            except Exception as e:
                logger.warning(
                    f"[PyPIExtractionStep] Error on doc {getattr(doc,'id',None)}: {e}"
                )
                self.stat_update("error")
                continue


def main():
    """
    Build a comprehensive in-memory repo with various edge cases:
      src/pkg/c.py         (no deps)
      src/pkg/b.py         (imports pkg.c)
      src/pkg/a.py         (imports pkg.b)
      tests/test_utils.py  (no deps)
      tests/test_a.py      (imports tests.test_utils)
      tests/test_meta.py   (imports tests.test_utils)
      src/pkg/d.py         (imports pkg.a, creating a circular dependency with a.py)
      native/add.cpp       (non-Python code)
      README.md            (non-code, included in dir_structure)
    Run the step and assert the topological order.
    """

    sample_payload = {
        "id": "example/repo_with_edge_cases",
        "file_contents": {
            "/src/pkg/__init__.py": "",
            "/src/pkg/c.py": "def c():\n    print('c')\n",
            "/src/pkg/b.py": "from pkg.c import c\n",
            "/src/pkg/a.py": "from pkg.b import c\n",  # a.py imports b.py
            "/src/pkg/d.py": "from pkg.a import c\n",  # circular dep: d -> a -> b -> c
            "/tests/test_utils.py": "def helper():\n    pass\n",
            "/tests/test_a.py": "from tests.test_utils import helper\nfrom pkg.a import c\n",
            "/tests/test_meta.py": "from tests.test_utils import helper\n",
            "/native/add.cpp": "int add(int a,int b){return a+b;}\n",
            "/README.md": "# Sample\n",
        },
        "metadata": {
            "repo_url": "https://example.org/repo",
            "snapshot_id": "deadbeef",
        },
    }

    inp = Document(
        text=sample_payload["file_contents"],
        id=sample_payload["id"],
        metadata=sample_payload["metadata"],
    )

    # Initialize your extractor step
    step = StackV2Extractor()

    outputs = list(step.run([inp]))
    assert len(outputs) == 1, "Expected exactly one output document"

    out = json.loads(outputs[0].text)

    # Expected topological order (python_code)
    # We have circular dependencies in a.py, b.py, d.py, so they should be properly handled
    expected_py = [
        'src/pkg/__init__.py',
        'src/pkg/c.py',
        'src/pkg/b.py',
        'src/pkg/a.py',
        'src/pkg/d.py',
    ]
    actual_py = list(out["python_code"].keys())
    assert (
        actual_py == expected_py
    ), f"Python topo order mismatch: {actual_py} != {expected_py}"

    # Expected topological order among tests
    # tests/test_utils.py has no deps; test_a.py & test_meta.py both depend on test_utils.py.
    # After removing test_utils, lexicographic tie-breaker => test_a.py then test_meta.py
    expected_tests = [
        "tests/test_utils.py",
        "tests/test_a.py",
        "tests/test_meta.py",
    ]
    actual_tests = list(out["python_tests_code"].keys())
    assert (
        actual_tests == expected_tests
    ), f"Test topo order mismatch: {actual_tests} != {expected_tests}"

    # Extra language map present & sorted (cpp code should be present)
    assert "cpp_code" in out and list(out["cpp_code"].keys()) == [
        "native/add.cpp"
    ], "cpp_code missing or wrong"

    # Metadata passthrough & dir structure present
    assert out["metadata"]["repo_url"] == "https://example.org/repo"
    assert "root" in out["dir_structure"]


if __name__ == "__main__":
    main()

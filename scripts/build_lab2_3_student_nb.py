"""Stub the afternoon solutions notebook into the student variant.

The afternoon notebook ``notebooks/lab2_lab3_nano_r1.ipynb`` carries the
SOLUTIONS (full bodies) for the 6 student-fillable functions:

    1. compute_reward
    2. compute_group_relative_advantages
    3. create_training_episodes (only the per-group loop body is stubbed)
    4. compute_kl_penalty
    5. compute_policy_loss_term
    6. compute_entropy

Each carries a ``# === STUDENT TODO === #`` sentinel placed immediately
before the body that students must fill in. This script reads the
solutions notebook, finds every code cell containing the sentinel,
replaces everything FROM the sentinel line through the end of the
function (or, for ``create_training_episodes``, through the
``return episodes, stats`` line) with::

    # === STUDENT TODO === #
    raise NotImplementedError("Fill me in!")

…and writes the result to
``notebooks/lab2_lab3_nano_r1_student.ipynb``.

This mirrors the morning notebook's stub mechanism (see
``scripts/build_nb.py``), but for an existing hand-authored notebook —
no nbformat regeneration of un-stubbed cells.

Usage::

    python scripts/build_lab2_3_student_nb.py
    python scripts/build_lab2_3_student_nb.py --check   # exit 1 if outputs are stale
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import List

import nbformat as nbf

REPO = Path(__file__).resolve().parent.parent
SOLUTIONS_NB = REPO / "notebooks" / "lab2_lab3_nano_r1.ipynb"
STUDENT_NB = REPO / "notebooks" / "lab2_lab3_nano_r1_student.ipynb"

SENTINEL = "# === STUDENT TODO === #"
STUB_BODY = 'raise NotImplementedError("Fill me in!")'


def _stub_cell_source(src: str) -> str:
    """Replace the post-sentinel body with NIE.

    The sentinel sits at some indentation level (always 4 spaces inside a
    function body in our notebooks). We:
      * keep everything up to and including the sentinel line,
      * replace the rest with `raise NotImplementedError("Fill me in!")` at
        the SAME indentation.

    This works for both:
      - Functions whose entire body is a TODO (compute_reward,
        compute_group_relative_advantages, the 3 PG sub-pieces).
      - ``create_training_episodes``, where only the per-group loop is
        stubbed; everything before the sentinel (signature, docstring,
        asserts, groups computation, output-list initialization) stays.

    For the create_training_episodes case, the trailing
    ``return episodes, stats`` is restored after the stub body so the
    function still type-checks and returns the expected (episodes, stats)
    shape — students will typically populate those vars and re-add the
    return.
    """
    if SENTINEL not in src:
        return src

    # Split on the FIRST sentinel only (defense against a stray sentinel
    # in a comment elsewhere — none in our notebooks today, but cheap).
    head, tail = src.split(SENTINEL, 1)

    # Detect indentation by looking at the LINE the sentinel sits on.
    # head ends with leading whitespace + "" then the sentinel begins.
    # Find the indentation of the line containing the sentinel:
    last_newline = head.rfind("\n")
    indent = head[last_newline + 1:] if last_newline != -1 else ""
    # indent is whitespace before "# === STUDENT TODO === #".

    # For create_training_episodes the function ends with a `return` line
    # AFTER the stubbed body. Detect by scanning the tail for the final
    # `return` at indent-level "    ".
    if "return episodes, stats" in tail:
        suffix = "\n\n    episodes = {\n        \"all_query_token_ids\": all_query_token_ids,\n        \"all_response_token_ids\": all_responses_token_ids,\n        \"all_advantages\": all_advantages,\n    }\n\n    return episodes, stats\n"
    else:
        suffix = "\n"

    return f"{head}{SENTINEL}\n{indent}{STUB_BODY}{suffix}"


def stub_notebook(solutions: nbf.NotebookNode) -> nbf.NotebookNode:
    """Return a deep-copy student variant of `solutions` with sentinels stubbed."""
    nb = copy.deepcopy(solutions)

    n_stubbed = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = "".join(cell.source) if isinstance(cell.source, list) else cell.source
        if SENTINEL not in src:
            continue
        cell.source = _stub_cell_source(src)
        n_stubbed += 1

    if n_stubbed != 6:
        raise RuntimeError(
            f"expected 6 sentinel'd cells, stubbed {n_stubbed}. "
            "Solutions notebook may be out of date."
        )

    # Hygiene: strip outputs/exec counts.
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []

    return nb


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="exit 1 if STUDENT_NB is stale relative to the current solutions",
    )
    args = parser.parse_args(argv)

    sol = nbf.read(SOLUTIONS_NB, as_version=4)
    stu = stub_notebook(sol)

    if args.check:
        existing = nbf.read(STUDENT_NB, as_version=4) if STUDENT_NB.exists() else None
        if existing is None:
            print(f"STALE: {STUDENT_NB} does not exist", file=sys.stderr)
            return 1
        # Compare by source-of-each-cell (ignore exec_count / outputs noise).
        a = [(c.cell_type, "".join(c.source) if isinstance(c.source, list) else c.source)
             for c in stu.cells]
        b = [(c.cell_type, "".join(c.source) if isinstance(c.source, list) else c.source)
             for c in existing.cells]
        if a != b:
            print(f"STALE: {STUDENT_NB} differs from current solutions stub.", file=sys.stderr)
            print(f"  Run: python scripts/build_lab2_3_student_nb.py", file=sys.stderr)
            return 1
        print(f"FRESH: {STUDENT_NB} matches solutions stub.")
        return 0

    nbf.write(stu, STUDENT_NB)
    print(f"Wrote {STUDENT_NB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

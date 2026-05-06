"""Round-trip test for ``scripts/build_nb.py``.

Verifies the generator emits structurally-valid notebooks with the
expected section count + TOC + STUDENT cells in both modes. We don't
run the notebooks here (that's nbmake's job in CI).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import nbformat as nbf
import pytest

REPO = Path(__file__).resolve().parent.parent
BUILD_SCRIPT = REPO / "scripts" / "build_nb.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_nb", BUILD_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_nb"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("mode", ["answer", "stub"])
def test_nb_round_trip(tmp_path: Path, mode: str) -> None:
    """Generator emits a valid notebook with all 12 section anchors."""
    builder = _load_builder()
    out = tmp_path / "nb.ipynb"
    rc = builder.main(["--out", str(out), "--mode", mode])
    assert rc == 0
    assert out.exists()

    nb = nbf.read(out, as_version=4)
    nbf.validate(nb)

    md_text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    for i in range(12):
        assert f'name="sec{i}"' in md_text, f"missing anchor sec{i}"

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    code_sources = "\n\n".join(c.source for c in code_cells)
    if mode == "stub":
        assert code_sources.count('raise NotImplementedError("Fill me in!")') == 5
        assert "Fill in the per-update step" in code_sources
    else:
        assert "Fill me in" not in code_sources
        assert "opt_demo.zero_grad()" in code_sources
        assert "opt_demo.step()" in code_sources

    # §6 transparent training-loop cell present in both modes.
    assert "N_UPDATES_INLINE" in code_sources
    assert "policy_demo" in code_sources

    # §0 calls reset() for clean state on fresh kernel.
    assert "_rl_reset(verbose=False)" in code_sources

    # §11 embeds the POMDP detour gif.
    assert "v7_pomdp_tr_detour.gif" in md_text

    # Total cell count is bounded.
    assert 30 <= len(nb.cells) <= 60, f"unexpected cell count: {len(nb.cells)}"


def test_default_builds_both_notebooks(tmp_path: Path, monkeypatch) -> None:
    """Running with no args emits both student.ipynb + solutions.ipynb."""
    builder = _load_builder()
    nb_dir = tmp_path / "notebooks"
    monkeypatch.setattr(builder, "REPO", tmp_path)
    rc = builder.main([])
    assert rc == 0
    assert (nb_dir / "student.ipynb").exists()
    assert (nb_dir / "solutions.ipynb").exists()


def test_stub_function_keeps_signature_and_docstring() -> None:
    """Stubbed body keeps the signature + docstring + sentinel."""
    builder = _load_builder()
    src = builder.stub_student_function("compute_returns_to_go")
    assert src.startswith("def compute_returns_to_go(")
    assert "Returns G_t" in src
    assert "# === STUDENT TODO === #" in src
    assert 'raise NotImplementedError("Fill me in!")' in src
    assert "running = rewards" not in src

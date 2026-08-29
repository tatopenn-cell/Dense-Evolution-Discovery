"""
Tests for scripts/chunk_distributed_disk_experiment.py -- imports the real
script (runs the full experiment on import, same convention as this repo's
other tests) and checks its real module-level results.
"""
import importlib.util
import pathlib
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _import_script(name: str):
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp = _import_script("chunk_distributed_disk_experiment")


def test_forced_geometry_gives_four_chunks_of_two_qubits():
    assert exp.ram_num_chunks == 4
    assert exp.ram_m == 2


def test_ram_chunk_matches_unchunked_baseline():
    assert exp.ram_matches_baseline is True


def test_distributed_path_actually_ran_on_multiple_devices():
    # JAX's device count is fixed at its first initialization in this
    # whole pytest process -- if an earlier-collected test file imports
    # dense_evolution/jax first, this script's own XLA_FLAGS setting (set
    # before ITS import, but not necessarily before jax's real first
    # init) arrives too late, and only the 1 real device is ever seen.
    # Running this script alone (`python scripts/chunk_distributed_disk_experiment.py`)
    # always exercises the real distributed path -- verified directly,
    # see docs/chunk_distributed_disk_experiment.md's own reported numbers.
    if not exp.distributed_available:
        pytest.skip("JAX already initialized with 1 device by an earlier-imported "
                     "test module in this pytest session -- run this script alone "
                     "to exercise the real distributed path.")
    assert exp.distributed_probs is not None
    assert exp.distributed_matches_baseline is True


def test_disk_overflow_path_matches_baseline():
    assert exp.disk_matches_baseline is True


def test_stride_pairs_cover_both_chunk_select_qubits():
    qubits_seen = {q for _, _, q in exp.stride_pairs}
    assert qubits_seen == {0, 1}
    assert len(exp.stride_pairs) == 4


def test_diagrams_were_actually_generated():
    assert (exp.ASSETS_DIR / "circuit.png").exists()
    assert (exp.ASSETS_DIR / "chunk_layout.png").exists()
    assert (exp.ASSETS_DIR / "disk_layout.png").exists()

"""scripts/print_circuits.py runs and prints what the training circuit actually is."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "print_circuits.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("print_circuits", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_prints_both_ansatze_at_the_nominal_config(script, capsys) -> None:
    assert script.main([]) == 0
    out = capsys.readouterr().out
    assert "SIMPLE ansatz | n_qb=5, n_l=2 | readout=dual" in out
    assert "BEL ansatz | n_qb=5, n_l=2 | readout=dual" in out
    assert "trainable params : 110" in out
    assert "paper states 110 trainable parameters -> matches" in out


def test_drawn_circuit_has_one_row_per_qubit_and_the_controlled_ring(script, capsys) -> None:
    script.main(["--ansatz", "bel", "--n-qubits", "4", "--n-layers", "1"])
    out = capsys.readouterr().out
    rows = [line for line in out.splitlines() if line[:2] in {"0:", "1:", "2:", "3:"}]
    assert len(rows) == 4
    assert "╭●" in out or "╰●" in out


def test_provenance_is_always_printed(script, capsys) -> None:
    script.main(["--ansatz", "simple"])
    assert "UNVERIFIED" in capsys.readouterr().out
    script.main(["--ansatz", "bel"])
    assert "PROVISIONAL" in capsys.readouterr().out


def test_qasm_flag_emits_parseable_qasm(script, capsys) -> None:
    openqasm3 = pytest.importorskip("openqasm3")
    script.main(["--ansatz", "bel", "--n-qubits", "3", "--n-layers", "1", "--qasm"])
    out = capsys.readouterr().out
    qasm = out[out.index("OPENQASM 3.0;") :].split("\n\n")[0]
    openqasm3.parse(qasm)


def test_paper_config_table(script, capsys) -> None:
    assert script.main(["--paper-configs"]) == 0
    out = capsys.readouterr().out
    assert "55/110" in out and "0/11 rows entered" in out
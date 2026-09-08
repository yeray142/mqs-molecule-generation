"""Tests for the SMILES tokenizer (paper §2.2.1)."""

from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem, RDLogger

from mqs_molecule_generation.data.tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    SmilesTokenizer,
)

RDLogger.DisableLog("rdApp.*")

# Real, RDKit-canonicalized molecules covering the SMILES features that
# actually matter: rings, branches, stereochemistry, E/Z bonds, aromatic
# heterocycles, isotopes, fused/spiro systems, and a two-digit ring closure.
_RAW_TEST_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CN1CCC[C@H]1c1cccnc1",  # nicotine (stereo)
    "C/C=C/C",  # E-alkene
    r"C/C=C\C",  # Z-alkene
    "CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O",  # steroid-like, many ring closures
    "O=C(O)c1ccccc1Nc1ccccc1Cl",  # diclofenac-like
    "c1ccc2[nH]ccc2c1",  # indole
    "CC(C)(C)OC(=O)N1CCC(N)CC1",  # boc-piperidine
    "[13CH4]",  # isotope
    "CC(=O)N[C@@H](CC1=CC=CC=C1)C(=O)O",  # N-acetylphenylalanine
    "C1CCC2(CC1)CCCCC2",  # spiro
    "C1CC2CCC3CCC4CCC5CCC6CCC7CCC8CCC9CCC%10CCC1C2C3C4C5C6C7C8C9%10",  # 2-digit ring closure
]


def _canonical(smiles_list: list[str]) -> list[str]:
    out = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"fixture SMILES failed to parse: {smi!r}"
        out.append(Chem.MolToSmiles(mol))
    return out


CANONICAL_TEST_SMILES = _canonical(_RAW_TEST_SMILES)


class TestFromData:
    def test_vocab_size_matches_idx_to_char_length(self) -> None:
        # The bug this design eliminates: with a hand-curated vocab, a
        # duplicated character can desync these two -- here they cannot,
        # because the vocabulary is built from a set.
        tok = SmilesTokenizer.from_data(CANONICAL_TEST_SMILES)
        assert tok.vocab_size == len(tok.idx_to_char)

    def test_every_index_is_reachable(self) -> None:
        # Every index in [0, vocab_size) must decode to something -- no dead
        # slots, no gaps.
        tok = SmilesTokenizer.from_data(CANONICAL_TEST_SMILES)
        assert set(tok.idx_to_char.keys()) == set(range(tok.vocab_size))

    def test_char_to_idx_and_idx_to_char_are_inverses(self) -> None:
        tok = SmilesTokenizer.from_data(CANONICAL_TEST_SMILES)
        for char, idx in tok.char_to_idx.items():
            assert tok.idx_to_char[idx] == char

    def test_covers_every_character_in_training_data(self) -> None:
        tok = SmilesTokenizer.from_data(CANONICAL_TEST_SMILES)
        all_chars = set("".join(CANONICAL_TEST_SMILES))
        assert all_chars <= set(tok.char_to_idx.keys())

    def test_special_tokens_present_and_distinct(self) -> None:
        tok = SmilesTokenizer.from_data(CANONICAL_TEST_SMILES)
        indices = {tok.pad_idx, tok.unk_idx, tok.bos_idx, tok.eos_idx}
        assert len(indices) == 4

    def test_empty_data_gives_vocab_of_only_specials(self) -> None:
        tok = SmilesTokenizer.from_data([])
        assert tok.vocab_size == 4


class TestEncodeDecode:
    def test_basic_roundtrip(self) -> None:
        tok = SmilesTokenizer.from_data(CANONICAL_TEST_SMILES)
        smi = CANONICAL_TEST_SMILES[0]
        assert tok.decode(tok.encode(smi)) == smi

    def test_unknown_character_maps_to_unk(self) -> None:
        tok = SmilesTokenizer.from_data(["CCO"])
        # 'X' never appeared in training data.
        ids = tok.encode("CXO")
        assert ids[1] == tok.unk_idx

    def test_add_bos_eos(self) -> None:
        tok = SmilesTokenizer.from_data(["CCO"])
        ids = tok.encode("CCO", add_bos=True, add_eos=True)
        assert ids[0] == tok.bos_idx
        assert ids[-1] == tok.eos_idx
        assert len(ids) == len("CCO") + 2

    def test_decode_strips_special_tokens_by_default(self) -> None:
        tok = SmilesTokenizer.from_data(["CCO"])
        ids = tok.encode("CCO", add_bos=True, add_eos=True)
        assert tok.decode(ids) == "CCO"

    def test_decode_can_keep_special_tokens(self) -> None:
        tok = SmilesTokenizer.from_data(["CCO"])
        ids = tok.encode("CCO", add_bos=True, add_eos=True)
        kept = tok.decode(ids, strip_special=False)
        assert kept == f"{BOS_TOKEN}CCO{EOS_TOKEN}"

    def test_empty_string_roundtrips(self) -> None:
        tok = SmilesTokenizer.from_data(["CCO"])
        assert tok.decode(tok.encode("")) == ""


class TestRoundtripOnRepresentativeMolecules:
    """Real, RDKit-canonicalized molecules -- rings, stereo, E/Z, isotopes,
    fused/spiro systems, two-digit ring closures. Direct check of the
    docstring's stated design goal.
    """

    @pytest.fixture
    def tok(self) -> SmilesTokenizer:
        return SmilesTokenizer.from_data(CANONICAL_TEST_SMILES)

    @pytest.mark.parametrize("smiles", CANONICAL_TEST_SMILES)
    def test_roundtrip(self, tok: SmilesTokenizer, smiles: str) -> None:
        decoded, ok = tok.roundtrip(smiles)
        assert ok, f"roundtrip failed: {smiles!r} -> {decoded!r}"


class TestAgainstRealTrainSet:
    """The actual stated gate: lossless round-trip on 100% of the train set."""

    @pytest.mark.slow
    def test_full_train_set_roundtrip(self) -> None:
        train_path = Path("data/processed/train.smi")
        if not train_path.exists():
            pytest.skip(f"{train_path} not found -- run scripts/prepare_data.py prepare first")

        train_smiles = train_path.read_text().split()
        tok = SmilesTokenizer.from_file(train_path)

        failures = []
        for smi in train_smiles:
            decoded, ok = tok.roundtrip(smi)
            if not ok:
                failures.append((smi, decoded))

        assert not failures, (
            f"{len(failures)}/{len(train_smiles)} molecules failed to round-trip. "
            f"First failure: {failures[0][0]!r} -> {failures[0][1]!r}"
        )
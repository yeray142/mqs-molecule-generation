"""Character-level SMILES tokenizer (paper §2.2.1, data-driven vocabulary).

Deliberately does NOT hand-enumerate SMILES grammar into a fixed vocabulary
constant. MOSES's own tokenizer (``moses/utils.py::CharVocab``), which the
paper's VAE architecture is taken from wholesale, doesn't either -- it builds
the vocabulary as the union of every character that actually appears in the
training data:

    chars = set()
    for string in data:
        chars.update(string)

That is strictly more robust than a hand-written character list: it can't
omit a character the real data needs, and it can't include a stray one that
was never valid. A fixed-vocabulary version of this file was tried first and
had exactly these two failure modes -- a missing '$' (quadruple bond) despite
being listed in its own docstring, plus 'd'/'e'/'^' additions that aren't
part of the OpenSMILES aromatic-atom set (b c n o p s) -- alongside an
unrelated but more serious bug: two duplicated characters ('B', 'C') desynced
its vocab_size from the actual token index range, which would have crashed
the first time Phase 1's embedding layer saw one of those tokens. This
version has no equivalent failure mode because there is nothing to hand-curate.

Special tokens follow MOSES's own naming and ordering (``SpecialTokens`` in
moses/utils.py): pad, unknown, beginning/end of sequence, appended AFTER the
sorted data characters rather than before.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

_SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)


@dataclass(frozen=True)
class SmilesTokenizer:
    """Character-level SMILES tokenizer with a data-driven vocabulary."""

    char_to_idx: dict[str, int]
    idx_to_char: dict[int, str]

    @classmethod
    def from_data(cls, smiles_list: list[str]) -> SmilesTokenizer:
        """Build the vocabulary from the union of characters in ``smiles_list``.

        Mirrors ``moses.utils.CharVocab.from_data``: sorted data characters
        first, special tokens appended after.

        Raises:
            ValueError: If any special token string is found among the
                per-character data (matches MOSES's own guard). In practice
                unreachable through this method's ``list[str]`` interface:
                ``chars.update(smiles)`` decomposes each string into
                individual characters, and a single character can never
                equal a 4+ character token like '<pad>'. Kept anyway,
                defensively, for parity with MOSES's own implementation and
                in case this is ever called with pre-split character lists.
        """
        chars: set[str] = set()
        for smiles in smiles_list:
            chars.update(smiles)

        overlap = chars & set(_SPECIAL_TOKENS)
        if overlap:
            raise ValueError(f"Special token(s) found in data characters: {overlap}")

        all_symbols = sorted(chars) + list(_SPECIAL_TOKENS)
        char_to_idx = {c: i for i, c in enumerate(all_symbols)}
        idx_to_char = dict(enumerate(all_symbols))
        return cls(char_to_idx=char_to_idx, idx_to_char=idx_to_char)

    @classmethod
    def from_file(cls, path: Path) -> SmilesTokenizer:
        """Build from a frozen ``.smi`` file (one SMILES per line)."""
        smiles_list = path.read_text().split()
        return cls.from_data(smiles_list)

    def save(self, path: Path) -> None:
        """Persist the exact vocabulary as JSON.

        The vocabulary is data-dependent (built from whatever SMILES
        ``from_data`` saw), so a model checkpoint's embedding weights only
        line up with the RIGHT characters if the SAME char_to_idx mapping is
        used to reload it. Re-deriving the tokenizer from a `.smi` file later
        is only safe if that file is byte-identical to what was used at
        training time -- saving it alongside the checkpoint removes that
        fragile assumption entirely.
        """
        path.write_text(json.dumps(self.char_to_idx))

    @classmethod
    def load(cls, path: Path) -> SmilesTokenizer:
        """Load a vocabulary saved by :meth:`save`."""
        char_to_idx = json.loads(path.read_text())
        idx_to_char = {i: c for c, i in char_to_idx.items()}
        return cls(char_to_idx=char_to_idx, idx_to_char=idx_to_char)

    def encode(self, smiles: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Convert a SMILES string to a list of token indices.

        Unknown characters map to the UNK token rather than raising, so a
        single out-of-vocabulary molecule can't abort a batch encode.
        """
        ids = [self.char_to_idx.get(c, self.char_to_idx[UNK_TOKEN]) for c in smiles]
        if add_bos:
            ids = [self.bos_idx, *ids]
        if add_eos:
            ids = [*ids, self.eos_idx]
        return ids

    def decode(self, tokens: list[int], strip_special: bool = True) -> str:
        """Convert token indices back to a SMILES string.

        Args:
            tokens: Token indices to decode.
            strip_special: If True (default), BOS/EOS/PAD tokens are dropped
                from the output rather than rendered as literal '<bos>' etc.
                UNK is always rendered as-is (there's no way to recover the
                original character from it).
        """
        special = {self.bos_idx, self.eos_idx, self.pad_idx} if strip_special else set()
        return "".join(
            self.idx_to_char[t] for t in tokens if t not in special and t in self.idx_to_char
        )

    def roundtrip(self, smiles: str) -> tuple[str, bool]:
        """Encode then decode (no BOS/EOS); returns (decoded_smiles, success)."""
        decoded = self.decode(self.encode(smiles), strip_special=False)
        return decoded, decoded == smiles

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_idx)

    @property
    def pad_idx(self) -> int:
        return self.char_to_idx[PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.char_to_idx[UNK_TOKEN]

    @property
    def bos_idx(self) -> int:
        return self.char_to_idx[BOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.char_to_idx[EOS_TOKEN]
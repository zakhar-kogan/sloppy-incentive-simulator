from __future__ import annotations

from typing import Protocol

from icframe.domain.norms import LawProgram


class CodifierPort(Protocol):
    def codify(self, prompt: str) -> LawProgram:
        """Translate a human prompt into a typed law program."""

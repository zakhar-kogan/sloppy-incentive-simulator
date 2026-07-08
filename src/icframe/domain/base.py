from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict

Scalar = str | int | float | bool


class ICFrameModel(BaseModel):
    """Strict base model for all boundary objects."""

    model_config = ConfigDict(strict=True, extra="forbid", validate_assignment=True)

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json", exclude_none=True), sort_keys=True)

from __future__ import annotations

import pytest
from pydantic import ValidationError

from icframe.ui.request_models import PaginationQuery, RunRequest, StudyRequest


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"limit": 0}, "limit"),
        ({"limit": -1}, "limit"),
        ({"limit": 501}, "limit"),
        ({"offset": -1}, "offset"),
    ],
)
def test_pagination_is_bounded(payload, field) -> None:
    with pytest.raises(ValidationError, match=field):
        PaginationQuery.model_validate(payload)


def test_run_seed_batch_is_bounded() -> None:
    with pytest.raises(ValidationError, match="at most 100"):
        RunRequest.model_validate({"pack": "public_goods", "seeds": list(range(101))})


def test_study_seed_batch_is_bounded() -> None:
    with pytest.raises(ValidationError, match="at most 100"):
        StudyRequest.model_validate({"pack": "public_goods", "seeds": list(range(101))})

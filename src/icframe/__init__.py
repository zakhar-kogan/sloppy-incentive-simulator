"""ICFRAME v0.4 canonical public API."""

from .core import (
    CompilationError,
    LoadedDomainPack,
    RuntimeEngine,
    RuntimePlan,
    compile_runtime,
    list_domain_packs,
    load_domain_pack,
    run_experiment,
)
from .domain.incentive_spec import IncentiveSpec, load_incentive_spec
from .domain.run import ParameterRange, RunConfig, RunSummary, StudyConfig, StudySummary
from .llm import LiteLLMClient
from .replay import replay_run
from .study import run_study
from .version import __version__

__all__ = [
    "CompilationError",
    "IncentiveSpec",
    "LiteLLMClient",
    "LoadedDomainPack",
    "ParameterRange",
    "RunConfig",
    "RunSummary",
    "RuntimeEngine",
    "RuntimePlan",
    "StudyConfig",
    "StudySummary",
    "__version__",
    "compile_runtime",
    "list_domain_packs",
    "load_domain_pack",
    "load_incentive_spec",
    "replay_run",
    "run_experiment",
    "run_study",
]

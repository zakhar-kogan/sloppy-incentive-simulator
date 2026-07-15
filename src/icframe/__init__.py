"""ICFRAME v0.5 canonical public API."""

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
from .domain.run import (
    AgentStatistics,
    ExecutionProvenance,
    LLMUsageSummary,
    ParameterRange,
    PlannerKind,
    RunConfig,
    RunStatus,
    RunSummary,
    StudyConfig,
    StudySummary,
)
from .llm import LiteLLMClient
from .orchestration import JobHandle, cancel_job, get_job, submit_run, submit_study, sync_jobs
from .planning import MatrixPlanner, OptunaPlanner, RandomPlanner, StudyPlan, TrialSpec
from .profiles import ProfileRegistry, load_profiles
from .replay import replay_run
from .study import run_study
from .version import __version__

__all__ = [
    "AgentStatistics",
    "CompilationError",
    "ExecutionProvenance",
    "IncentiveSpec",
    "JobHandle",
    "LLMUsageSummary",
    "LiteLLMClient",
    "LoadedDomainPack",
    "MatrixPlanner",
    "OptunaPlanner",
    "ParameterRange",
    "PlannerKind",
    "ProfileRegistry",
    "RandomPlanner",
    "RunConfig",
    "RunStatus",
    "RunSummary",
    "RuntimeEngine",
    "RuntimePlan",
    "StudyConfig",
    "StudyPlan",
    "StudySummary",
    "TrialSpec",
    "__version__",
    "cancel_job",
    "compile_runtime",
    "get_job",
    "list_domain_packs",
    "load_domain_pack",
    "load_incentive_spec",
    "load_profiles",
    "replay_run",
    "run_experiment",
    "run_study",
    "submit_run",
    "submit_study",
    "sync_jobs",
]

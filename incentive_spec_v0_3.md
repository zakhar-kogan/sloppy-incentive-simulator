# IncentiveSpec v0.3

A domain-neutral intermediate representation for configurable incentive-system simulations.

This spec is intended for coding agents. It defines the core ontology, runtime expectations, TOML scenario format, observability requirements, and implementation constraints for a simulator that can express KPI gaming, reward-model hacking, tokenomics attacks, and mixed human-agent governance systems.

Version v0.3 is the v2 implementation target. Version v0.2 remains the stable implemented baseline.

## 0. Design position

This project is not a replacement for PettingZoo, Gymnasium, Mesa, NetLogo, or other simulation/MARL frameworks.

It is an incentive-system specification layer and small runtime that can later export or adapt into those ecosystems.

Core idea:

```text
IncentiveSpec TOML
  -> validated domain-neutral IR
  -> runtime simulation with observation/policy/constraint instrumentation
  -> event logs / metrics / plots / replay artifacts
  -> optional adapters: NetworkX, PettingZoo, MO-style vector reward, Mesa-style ABM, LLM policy runtimes
```

The first serious domain pack should be organizational KPI / tokenmaxxing, but the core must not contain hardcoded concepts like tokens, slop, LOC, tickets, defects, or KPIs.

## 1. Core abstractions

### 1.1 Domain-neutral kernel

The kernel consists of:

```text
state stores
agents
actions
transitions
outcome vectors
agent scalarizers
observation / visibility model
governance and enforcement model
memory / learning model
policy decision log
observability / replay model
event log
metrics
```

### 1.1.1 v0.3 additions

v0.3 adds these first-class subsystems:

```text
policy backend interface
bandit / learning policy state
LLM client abstraction
LLM call log
constraint explanation log
OpenTelemetry-shaped observability events
replay manifest
PettingZoo / MARL adapters
```

The core runtime remains the source of mechanics. External adapters may choose actions,
serve agents, train policies, export traces, or explain constraints, but they must not
silently mutate transition semantics.

### 1.2 State/action/transition primitive

The canonical graph primitive is:

```text
state -> action -> next_state
```

Each transition has typed attributes:

```text
preconditions
probabilities
effects
conditional effects
enforcement
governance metadata
observation / visibility
prompt descriptions
state updates
tags
```

Do not make different graph edge kinds semantically required. Use one transition schema plus optional typed blocks and tags.

### 1.3 Outcome vector

The runtime always computes an `OutcomeVector`:

```python
OutcomeVector = dict[str, float]
```

Outcome channel names are declared by the domain pack and should be namespaced strings:

```text
observed.*
latent.*
governance.*
agent.*
social.*
market.*
custom_namespace.*
```

Examples:

```text
observed.kpi_score
observed.activity_metric
latent.goal_value
latent.safety_margin
governance.audit_cost
governance.sanction_cost
agent.personal_payoff
social.trust_delta
market.liquidity_quality
```

The core does not know what these channels mean. Metrics, scalarizers, and plots interpret them.

### 1.4 Scalarization

Agents do not need to optimize the full outcome vector directly. Each agent or archetype has a scalarizer:

```text
scalar_reward_i = sum(weight_i[channel] * outcome[channel])
```

The runtime must store vector outcomes even if v1 agents only consume scalar rewards. This keeps later MORL/vector-reward support from becoming architecture-breaking.

## 2. State model

The runtime should maintain multiple state stores.

```text
global_state:      simulation-level variables
agent_state:       per-agent variables
object_state:      per-task / per-resource / per-contract variables
relation_state:    pairwise or network variables between agents/objects
graph_state:       dynamic graph-level variables, if needed
memory_state:      per-agent discovered knowledge and learned estimates
```

### 2.1 Global state

Examples:

```text
round
budget_remaining
aggregate_activity
institutional_trust
market_price
network_health
```

### 2.2 Agent state

Examples:

```text
role
archetype
reputation
wealth
rule_following_propensity
exploit_search_propensity
known_states
known_transitions
learned_action_values
beliefs_about_sanctions
```

### 2.3 Object/task/resource state

Examples:

```text
task.difficulty
task.hidden_value
task.status
contract.risk
resource.stock
proposal.quality
```

### 2.4 Relation/link state

Examples:

```text
trust(agent_a, agent_b)
communication_edge(agent_a, agent_b)
collusion_edge(agent_a, agent_b)
reputation_report(agent_a, agent_b)
```

### 2.5 Transition-local state updates

Transitions may update any state store through a typed `state_updates` block.

Example:

```toml
[transitions.state_updates.global.add]
"global.aggregate_activity" = 1.0

[transitions.state_updates.actor.add]
"agent.reputation" = -0.1

[transitions.state_updates.task.set]
"task.status" = "reported_complete"
```

## 3. Population-specific effects and rewards

Yes: transitions may have population-specific effects, rewards, sanctions, probabilities, and visibility.

The recommended pattern is:

```text
base transition effect
+ selector-based conditional overlays
+ agent scalarizer
```

Do not duplicate the transition for every population unless the action is semantically different.

### 3.1 Base effects

Base effects apply to all actors unless overridden or augmented.

```toml
[transitions.effects]
"observed.activity_metric" = 1.0
"latent.goal_value" = 0.2
"agent.personal_payoff" = 1.0
```

### 3.2 Conditional effects

Conditional effects apply when a selector matches the actor, target, task, relation, or global state.

Supported operations:

```text
add       add to base value
multiply  multiply existing value
set       replace existing value
```

Selector dimensions:

```text
actor.population
actor.archetype
actor.role
actor.attributes.*
target.population
target.role
task.type
task.attributes.*
global.*
relation.*
```

### 3.3 Example: government actor receives higher fine

```toml
[[transitions]]
id = "misreport_activity"
from = "working"
action = "misreport_activity"
to = "reported_high_activity"
availability = "possible_violation"
norm_status = "forbidden"
tags = ["proxy_gain", "auditable"]

[transitions.effects]
"observed.activity_metric" = 10.0
"latent.goal_value" = -1.0
"agent.personal_payoff" = 3.0

[transitions.enforcement]
audit_probability = 0.25
detection_probability = 0.60

[transitions.enforcement.sanction_if_detected]
"agent.personal_payoff" = -5.0
"governance.fine_collected" = 5.0

[[transitions.conditional_effects]]
priority = 10
operation = "add"

[transitions.conditional_effects.selector.actor]
role = ["governmental_actor", "regulated_actor"]

[transitions.conditional_effects.effects_if_detected]
"agent.personal_payoff" = -20.0
"governance.fine_collected" = 20.0
"social.trust_delta" = -0.5
```

Interpretation:

- Everyone gets the base effect if they misreport.
- If detected, everyone gets the base sanction.
- Governmental or regulated actors receive an additional sanction and trust loss.
- Their scalarizer then determines how strongly this affects action choice.

### 3.4 Actor, target, observer effects

Some actions affect multiple parties. The runtime should allow scoped effects:

```text
actor effects
target effects
observer effects
global effects
relation effects
```

Example:

```toml
[transitions.effects.actor]
"agent.personal_payoff" = 2.0
"agent.reputation_delta" = -0.1

[transitions.effects.target]
"agent.personal_payoff" = -1.0
"agent.trust_delta" = -0.2

[transitions.effects.global]
"latent.goal_value" = -0.5
"social.trust_delta" = -0.3
```

If v1 implementation is kept simple, flatten these into namespaced channels and include `actor_id`, `target_id`, and `scope` in the event log.

## 4. Rule hardness and enforcement

Rule status and physical availability must be separate.

### 4.1 Availability

```text
hard_available       action can be taken
hard_blocked         action cannot be taken by the runtime
possible_violation   action can be taken, but is a violation or risky behavior
```

### 4.2 Norm status

```text
permitted
forbidden
obligatory
discouraged
unknown
```

### 4.3 Enforcement

Enforcement is probabilistic and optionally noisy.

Fields:

```text
audit_probability
detection_probability
false_positive_probability
false_negative_probability
enforcement_probability
sanction_if_detected
reward_if_compliant
restorative_action
appeal_action
```

A forbidden action can still be physically possible if `availability = "possible_violation"`. A permitted action can still have bad latent effects. A rule can exist without being visible to all agents.

### 4.4 Social-law / constraint layer

v0.3 keeps mechanics in Python but expands the Clingo/ASP adapter into a richer
social-law layer.

The constraint layer may decide or explain:

```text
available(actor, state, action)
blocked(actor, state, action)
permitted(actor, state, action)
forbidden(actor, state, action)
discouraged(actor, state, action)
obligatory(actor, state, action)
violated(actor, state, action, rule_id)
requires_remediation(actor, action, remediation_action)
incompatible(action_a, action_b)
role_may_take(role, action)
```

It must not compute:

```text
outcome vectors
scalar rewards
audit/detection samples
enforcement samples
state mutations
agent learning updates
metrics
```

Constraint explanations are event-linked records:

```json
{
  "constraint_id": "constraint_abc",
  "policy_decision_id": "decision_123",
  "event_id": "event_456",
  "actor_id": "agent_07",
  "state": "working",
  "action": "misreport_activity",
  "available": true,
  "blocked": false,
  "norm_status": "forbidden",
  "obligations": [],
  "violations": ["truthful_reporting_rule"],
  "remediation_actions": ["remediate"],
  "explanation_facts": [
    "possible_violation",
    "norm_forbidden",
    "auditable",
    "remediation_available"
  ]
}
```

Richer policy-as-code generation, counterfactual law edits, and formal governance
analysis are v2 implementation work over this v0.3 contract.

## 5. Observation, visibility, and agent knowledge

Agents should see the system through a `visibility_profile`.

### 5.1 Graph visibility

```text
full_graph        agent sees all states/transitions defined for its role
local_graph       agent sees only current state and adjacent visible transitions
discovered_graph  agent starts local/empty and remembers explored transitions
prompt_only       agent sees only natural-language context and action labels
black_box         agent sees available action IDs only, no graph/reward details
none              mostly for non-learning scripted agents or hidden actors
```

### 5.2 Outcome/reward visibility

```text
full_numeric      sees numeric outcome channels
own_scalar        sees only scalar reward/payoff
sign_only         sees whether effect is positive/negative/neutral
ordinal           sees rank/order of actions, not magnitudes
noisy_numeric     sees perturbed estimates
label_only        sees words like "good", "bad", "risky", "unknown"
hidden            sees nothing directly
learned           starts hidden/noisy and updates beliefs from experience
```

Reward visibility should be separately configurable for:

```text
observed outcomes
latent outcomes
governance outcomes
sanctions
audit probabilities
detection probabilities
other agents' outcomes
```

### 5.3 Prompt visibility

Prompt descriptions are for LLM agents only. They should not directly change mechanics.

Recommended fields:

```toml
[transitions.prompt]
public = "Visible to agents under normal policy visibility."
actor_view = "Action-specific description for the acting agent."
auditor_view = "Description for auditors or evaluator agents."
hidden_designer_note = "Not shown to agents; used for docs/tests."
```

### 5.4 Discovery and memory

Agents with `graph_visibility = "discovered_graph"` maintain a memory state.

Minimum memory fields:

```text
known_states
known_transitions
visit_counts[action]
estimated_outcomes[action][channel]
estimated_scalar_reward[action]
estimated_audit_probability[action]
estimated_detection_probability[action]
observed_sanctions[action]
last_seen_events
```

Memory config:

```toml
[archetypes.explorer.memory]
enabled = true
mode = "discovered_graph"
max_events = 200
learn_transition_outcomes = true
learn_audit_probabilities = true
forgetting_rate = 0.01
```

### 5.5 Why local / hidden views are needed

The same simulator should support:

- transparent games where agents know the rules;
- realistic organizations where employees know only local workflows and vague sanctions;
- hidden-test / hidden-evaluator settings in reward modeling;
- tokenomics systems where participants infer incentives empirically;
- audits where exact detection probabilities are not public;
- asymmetric information between workers, auditors, managers, and adversaries;
- exploratory agents that learn the incentive landscape through trial and error.

Therefore full graph visibility should be supported, but not assumed.

## 6. Agent configuration

Use archetypes plus population overrides.

### 6.1 Archetype fields

```text
policy backend
role
scalarizer
behavior parameters
visibility profile
memory settings
LLM settings, if applicable
initial state
initial resources
```

### 6.2 Policy backends

Policy backends implement a shared decision contract:

```python
class PolicyBackend:
    def choose_action(
        self,
        observation: Observation,
        action_space: list[ActionCandidate],
        memory: MemoryState,
        rng: Random,
    ) -> PolicyDecision: ...
```

A policy decision must include:

```text
policy_decision_id
observation_id
agent_id
candidate_actions
chosen_action
target_id, if any
estimated_scalar_rewards, if available
estimated_outcome_vectors, if available
decision_probability or confidence, if available
rationale, if available
failure mode, if no valid action is chosen
policy_state_delta, if learning occurred
```

Supported v0.3 policy families:

```text
deterministic               symbolic fixed preference order
scripted                    scenario-defined action script
stochastic_weighted          probabilistic choice from scalarized estimates
epsilon_greedy_bandit        online bandit with exploration rate
ucb_bandit                   upper-confidence-bound bandit
thompson_sampling_bandit     posterior-sampling bandit
contextual_bandit            bandit using observation features
q_learning_simple            tabular local learner
pettingzoo_external          action supplied by external MARL environment wrapper
litellm_policy               LLM action selection through LiteLLM
agno_policy                  optional Agno agent/team/workflow adapter
llm_policy                   alias for configured default LLM policy adapter
```

Bandits are lightweight online-learning policies, not full MARL. They update
per-agent estimates from observed scalar rewards and/or visible outcome vectors.

MARL is an adapter target. The core should expose observations, action masks, and
scalar/vector rewards to PettingZoo-compatible environments, but training loops
and algorithm libraries remain outside the core.

### 6.3 Behavior parameters

Recommended generic fields:

```text
rule_following_propensity
exploit_search_propensity
sanction_sensitivity
risk_tolerance
information_sharing_propensity
deception_propensity
reputation_sensitivity
social_welfare_weight
learning_rate
exploration_rate
memory_length
```

Avoid making personality fields mechanically meaningful for non-LLM agents. Use measurable behavioral parameters instead.

### 6.4 LLM agent fields

```toml
[archetypes.llm_worker.llm]
backend = "litellm"
model = "openai/gpt-5.5-pro"
temperature = 0.3
max_context_events = 20
include_action_descriptions = true
include_visible_graph = true
include_visible_rewards = false
require_json_action = true
response_schema = "action_choice_v1"
system_prompt = """
You are an agent in a simulated incentive system. Choose one available action.
Use only information visible to your role.
"""
```

### 6.5 LLM policy adapters

LiteLLM is the default LLM call adapter for v0.3 because it provides a provider-neutral
completion interface, routing, fallbacks, retry behavior, token accounting, and cost hooks.

Agno is an optional higher-level adapter for experiments that need richer agent/team/workflow
or AgentOS-style runtime behavior. Agno should wrap IncentiveSpec observations and action
choices; it must not replace the IncentiveSpec observation compiler or transition runtime.

All LLM policy implementations must depend on an `LLMClient` interface:

```python
class LLMClient:
    def complete(self, request: LLMRequest) -> LLMResponse: ...
```

Required clients:

```text
FakeLLMClient       deterministic unit-test client
RecordedLLMClient   replays llm_calls.jsonl
LiteLLMClient       live provider calls through LiteLLM
AgnoPolicyAdapter   optional adapter for Agno agents, teams, or workflows
```

LLM action responses must be parsed as strict JSON:

```json
{
  "action": "token_bloat",
  "target_id": null,
  "reason": "The visible KPI reward appears highest."
}
```

Invalid JSON, hidden-information references, unavailable actions, or hard-blocked actions
must become logged policy failures. The runtime may fall back to a configured safe action,
skip the actor, or stop the run depending on experiment settings.

## 7. Observation compiler

The runtime must compile an agent-specific observation from the full state.

```python
def compile_observation(agent_id, full_state, spec):
    profile = visibility_profile(agent_id)
    visible_graph = filter_graph(full_graph, agent_id, profile)
    visible_actions = filter_actions(current_state, agent_id, profile)
    visible_outcomes = filter_outcomes(agent_id, profile)
    visible_sanctions = filter_enforcement(agent_id, profile)
    prompt_context = compile_prompt_context(agent_id, profile)
    memory = memory_state[agent_id]
    return Observation(...)
```

Observation must never expose hidden fields accidentally. All LLM prompts should be generated from the same observation compiler as symbolic agents.

## 8. Runtime step semantics

One simulation step:

```text
1. Create or continue a run/step context with stable trace IDs.
2. For each active agent, compile an observation and log observation metadata.
3. Policy chooses an action from available actions and logs a PolicyDecision.
4. Runtime checks availability, preconditions, and constraint explanations.
5. Runtime samples transition success/failure.
6. Runtime applies base effects.
7. Runtime applies conditional effects by selector and priority.
8. Runtime samples audits/detection/enforcement.
9. Runtime applies sanctions/rewards/restorative updates.
10. Runtime updates global, agent, object, relation, and memory state.
11. Runtime computes scalar rewards for each affected agent.
12. Runtime logs the full event, visible event projections, and links to observations,
    policy decisions, constraint explanations, and optional LLM calls.
13. Metrics are updated.
```

### 8.1 Scheduling modes

Support at least:

```text
sequential_fixed
sequential_random
parallel_simultaneous
staged
```

Staged scheduling example:

```text
workers act
peers review
auditors audit
managers update KPI regime
```

### 8.2 MARL / PettingZoo adapter

v0.3 should expose IncentiveSpec runs as PettingZoo-compatible environments.

Required adapter targets:

```text
PettingZoo AEC API        sequential / turn-based interaction
PettingZoo Parallel API   simultaneous interaction where the schedule allows it
```

The adapter must derive:

```text
agents              concrete population-expanded agent IDs
observations        from the shared observation compiler
action spaces       visible available actions
action masks        availability plus constraint checks
rewards             scalarized outcome vectors
infos               event IDs, constraint IDs, metric deltas, visible metadata
terminations        scenario-specific terminal states
truncations         experiment step limits
```

The core does not own MARL training loops. External libraries may train policies
against the adapter, then feed actions back through `pettingzoo_external`.

Vector-reward / MORL consumers should receive the full `OutcomeVector` in `infos`
or through a dedicated vector-reward adapter.

## 9. Metrics

Metrics should be configured from outcome channels and event fields.

Examples:

```toml
[metrics.goodhart_gap]
type = "difference"
proxy = "observed.kpi_score"
target = "latent.goal_value"
normalization = "zscore"

[metrics.exploit_rate]
type = "event_rate"
where_tags_include = ["exploit"]
denominator = "all_actions"

[metrics.governance_efficiency]
type = "ratio"
numerator = "metric.goodhart_gap_reduction"
denominator = "governance.audit_cost"
```

The core should provide common metric types:

```text
sum
mean
rate
difference
ratio
zscore_difference
rolling_mean
event_count
event_rate
```

## 10. Observability and replay

Observability is a first-class v0.3 subsystem. The goal is not just operational
monitoring; it is scientific reproducibility, auditability, and debuggability.

### 10.1 Required identifiers

Every run must create stable IDs:

```text
run_id                 unique run identifier
trace_id               trace-wide correlation ID
step_id                run-local step identifier
agent_id               concrete actor identifier
observation_id         observation snapshot identifier
policy_decision_id     policy decision identifier
constraint_id          constraint explanation identifier, when present
llm_call_id            LLM call identifier, when present
event_id               transition event identifier
metric_id              metric record identifier, when useful
```

Parent-child relationships:

```text
run
  step
    observation
      policy_decision
        llm_call, optional
      constraint_explanation, optional
      transition_event
        outcome_vector
        scalar_rewards
    metrics
```

Every transition event must be attributable to exactly one observation and one
policy decision unless the event is generated by the runtime itself.

### 10.2 Observability TOML section

```toml
[observability]
enabled = true
streams = [
  "events",
  "observations",
  "policy_decisions",
  "constraints",
  "llm_calls",
  "metrics",
  "memory"
]
artifact_dir = ".artifacts/runs"
jsonl = true
include_trace_ids = true
include_wall_time = true

[observability.redaction]
mode = "balanced"
prompt_capture = "hash_and_redacted"
llm_response_capture = "parsed_and_hash"
hidden_state_capture = "never"
hash_algorithm = "sha256"

[observability.replay]
enabled = true
record_rng_state = true
record_llm_calls = true
fail_on_missing_replay_call = true

[observability.exporters]
otel = false
langfuse = false
mlflow = false
litellm_callbacks = true
agno_tracing = false
```

Recommended redaction modes:

```text
full             store full prompts/responses and visible observations
balanced         store parsed outputs plus hashes and redacted prompts
hash_only        store hashes and metadata only
metadata_only    store IDs, model/provider, token/cost/latency only
```

Hidden latent outcomes must never appear in observations, prompts, or LLM request
bodies unless the actor visibility profile allows them.

### 10.3 Observability streams

Required artifacts:

```text
run_manifest.json
trace.jsonl
observations.jsonl
policy_decisions.jsonl
constraint_explanations.jsonl
llm_calls.jsonl
metrics.csv
agent_memory.json
```

`run_manifest.json` records:

```text
spec name and version
spec hash
run_id
trace_id
seed
runtime version
dependency versions
started_at / completed_at
artifact paths
redaction policy
replay policy
```

`policy_decisions.jsonl` records:

```text
policy_decision_id
observation_id
agent_id
policy backend
candidate actions
chosen action
target_id
estimated values
rationale or rationale hash
failure mode
policy state delta
```

`llm_calls.jsonl` records:

```text
llm_call_id
policy_decision_id
provider
model
request hash
response hash
parsed response
latency_ms
prompt_tokens
completion_tokens
total_tokens
estimated_cost
retry_count
fallback_used
error_type
redaction mode
```

### 10.4 Replay

Symbolic runs replay from:

```text
spec hash
seed
runtime version
dependency versions
```

LLM runs replay from:

```text
spec hash
seed
runtime version
llm_calls.jsonl
policy_decisions.jsonl
```

Recorded LLM calls are matched by `llm_call_id` or by a deterministic request hash.
Replay must fail clearly if a required recorded LLM call is absent and
`fail_on_missing_replay_call = true`.

## 11. TOML schema skeleton

```toml
[spec]
version = "0.3"
name = "example_incentive_system"
domain = "generic"

[experiment]
steps = 50
seeds = [1, 2, 3]
schedule = "sequential_random"

[observability]
enabled = true
streams = ["events", "observations", "policy_decisions", "constraints", "metrics"]
artifact_dir = ".artifacts/runs"
jsonl = true

[observability.redaction]
mode = "balanced"
prompt_capture = "hash_and_redacted"
llm_response_capture = "parsed_and_hash"
hidden_state_capture = "never"

[observability.replay]
enabled = true
record_rng_state = true
record_llm_calls = true

[outcome_space]
channels = [
  "observed.activity_metric",
  "latent.goal_value",
  "governance.audit_cost",
  "governance.fine_collected",
  "agent.personal_payoff",
  "social.trust_delta"
]

[states]
initial_global = "working"
all = ["working", "reported_high_activity", "audited", "sanctioned", "remediated"]

[actions]
all = ["real_work", "misreport_activity", "audit", "remediate"]

[visibility_profiles.full]
graph = "full_graph"
observed_outcomes = "full_numeric"
latent_outcomes = "full_numeric"
sanctions = "full_numeric"
audit_probabilities = "full_numeric"
prompts = true

[visibility_profiles.explorer]
graph = "discovered_graph"
observed_outcomes = "own_scalar"
latent_outcomes = "hidden"
sanctions = "learned"
audit_probabilities = "learned"
prompts = true

[visibility_profiles.prompt_only]
graph = "prompt_only"
observed_outcomes = "label_only"
latent_outcomes = "hidden"
sanctions = "label_only"
audit_probabilities = "hidden"
prompts = true

[archetypes.rule_follower]
policy = "stochastic_weighted"
role = "worker"
visibility_profile = "explorer"

[archetypes.rule_follower.scalarizer]
"observed.activity_metric" = 0.2
"latent.goal_value" = 1.0
"agent.personal_payoff" = 0.5
"governance.sanction_cost" = -1.0
"social.trust_delta" = 0.5

[archetypes.rule_follower.behavior]
rule_following_propensity = 0.9
exploit_search_propensity = 0.05
sanction_sensitivity = 0.8
risk_tolerance = 0.2
learning_rate = 0.05

[[population]]
archetype = "rule_follower"
count = 20

[[transitions]]
id = "real_work"
from = "working"
action = "real_work"
to = "working"
availability = "hard_available"
norm_status = "permitted"
tags = ["productive"]

[transitions.effects]
"observed.activity_metric" = 1.0
"latent.goal_value" = 1.0
"agent.personal_payoff" = 1.0
"social.trust_delta" = 0.05

[transitions.prompt]
public = "Do real work that moderately improves the activity metric and the underlying objective."

[[transitions]]
id = "misreport_activity"
from = "working"
action = "misreport_activity"
to = "reported_high_activity"
availability = "possible_violation"
norm_status = "forbidden"
tags = ["proxy_gain", "exploit", "auditable"]

[transitions.effects]
"observed.activity_metric" = 10.0
"latent.goal_value" = -1.0
"agent.personal_payoff" = 3.0
"social.trust_delta" = -0.1

[transitions.enforcement]
audit_probability = 0.25
detection_probability = 0.60
enforcement_probability = 1.0
restorative_action = "remediate"

[transitions.enforcement.sanction_if_detected]
"agent.personal_payoff" = -5.0
"governance.fine_collected" = 5.0
"governance.sanction_cost" = -5.0

[[transitions.conditional_effects]]
priority = 10
operation = "add"

[transitions.conditional_effects.selector.actor]
role = ["governmental_actor", "regulated_actor"]

[transitions.conditional_effects.effects_if_detected]
"agent.personal_payoff" = -20.0
"governance.fine_collected" = 20.0
"governance.sanction_cost" = -20.0
"social.trust_delta" = -0.5

[transitions.prompt]
public = "Report a high activity level that may not correspond to real progress."
actor_view = "This may improve your visible metric but could violate the rule if audited."
auditor_view = "Check whether reported activity corresponds to underlying progress."
```

## 12. Validation requirements

A v0.3 validator should check:

1. Every transition references declared states and actions.
2. Every outcome channel referenced by effects, scalarizers, and metrics is declared in `outcome_space.channels`.
3. Every population references a declared archetype.
4. Every archetype references a declared visibility profile.
5. Conditional-effect selectors reference valid dimensions.
6. Every metric references declared channels or previous metrics.
7. Prompt fields do not define mechanics directly.
8. Hard-blocked actions cannot be chosen by normal policies.
9. Possible-violation actions can be chosen but may trigger enforcement.
10. Hidden latent outcomes are not exposed by the observation compiler.
11. Seeds produce deterministic symbolic runs.
12. Observability streams reference known stream names.
13. Prompt and LLM response capture modes are valid redaction modes.
14. LLM policies define an LLM backend and response schema.
15. Recorded LLM replay settings are internally consistent.
16. Clingo/social-law rules do not define outcome mechanics.
17. PettingZoo adapters expose action masks consistent with runtime availability.

## 13. Event and decision log schemas

Each policy choice should produce a decision record. Each transition attempt should
produce an event linked to the relevant observation, decision, constraint explanation,
and optional LLM call.

Policy decision record:

```json
{
  "run_id": "run_001",
  "trace_id": "trace_001",
  "step": 12,
  "observation_id": "obs_abc",
  "policy_decision_id": "decision_def",
  "agent_id": "agent_07",
  "policy_backend": "epsilon_greedy_bandit",
  "candidate_actions": ["real_work", "token_bloat", "misreport_activity"],
  "chosen_action": "misreport_activity",
  "target_id": null,
  "estimated_scalar_rewards": {
    "real_work": 1.2,
    "token_bloat": 7.1,
    "misreport_activity": 8.2
  },
  "decision_probability": 0.86,
  "rationale": null,
  "llm_call_id": null,
  "failure_mode": null
}
```

LLM policy decision record should include `llm_call_id` and may store the rationale
as full text, parsed text, redacted text, or hash only depending on the observability
redaction policy.

```json
{
  "run_id": "run_001",
  "trace_id": "trace_001",
  "event_id": "event_xyz",
  "observation_id": "obs_abc",
  "policy_decision_id": "decision_def",
  "constraint_id": "constraint_jkl",
  "seed": 1,
  "step": 12,
  "actor_id": "agent_07",
  "actor_archetype": "proxy_maximizer",
  "actor_role": "worker",
  "target_id": null,
  "from_state": "working",
  "action": "misreport_activity",
  "to_state": "reported_high_activity",
  "availability": "possible_violation",
  "norm_status": "forbidden",
  "tags": ["proxy_gain", "exploit", "auditable"],
  "base_effects": {
    "observed.activity_metric": 10.0,
    "latent.goal_value": -1.0
  },
  "conditional_effects_applied": ["government_actor_extra_fine"],
  "audit_sampled": true,
  "detected": false,
  "enforced": false,
  "final_outcome_vector": {
    "observed.activity_metric": 10.0,
    "latent.goal_value": -1.0,
    "agent.personal_payoff": 3.0
  },
  "scalar_rewards": {
    "agent_07": 8.2
  },
  "visible_event_refs": {
    "agent_07": "visible_event_abc",
    "auditor_01": "visible_event_def"
  },
  "artifact_refs": {
    "observation": "observations.jsonl:obs_abc",
    "policy_decision": "policy_decisions.jsonl:decision_def",
    "constraint_explanation": "constraint_explanations.jsonl:constraint_jkl"
  }
}
```

## 14. Public interfaces

Required v0.3 public interfaces:

```python
load_incentive_spec(path) -> IncentiveSpec
run_incentive_simulation(spec, seed=None, observer=None) -> SimulationTrace
compile_observation(agent_id, full_state, spec) -> Observation
choose_action(policy, observation, action_space, memory, rng) -> PolicyDecision
validate_constraints(spec) -> ConstraintReport
explain_transition_availability(spec, actor, state, action) -> ConstraintExplanation
LLMClient.complete(request) -> LLMResponse
PettingZooIncentiveEnv(spec)
```

The `observer` argument receives structured lifecycle events and is responsible for
writing JSONL artifacts, exporting spans, or forwarding records to downstream systems.
The runtime should be correct when `observer=None`.

## 15. Test plan for v0.3 implementation

- v0.2 tokenmaxxing fixture still loads and runs.
- v0.3 fixture validates with observability config.
- JSONL artifacts are emitted and replayable.
- Policy decisions are logged for symbolic and bandit policies.
- Bandit memory updates are deterministic under fixed seeds.
- Mocked LLM policy chooses a valid action.
- Malformed LLM output is logged and rejected.
- Hidden latent outcomes do not appear in LLM prompts.
- Recorded LLM calls replay without network access.
- Clingo explanations include allowed, forbidden, blocked, obligatory, and violated facts.
- PettingZoo wrapper exposes action masks and scalar rewards correctly.
- Vector reward adapter exposes full outcome vectors without losing scalar rewards.

## 16. Implementation milestones

### v0.1

- TOML loader.
- Pydantic/dataclass schema validation.
- Namespaced outcome channels.
- State/action/transition graph.
- Base effects.
- Scalarizers.
- Deterministic and stochastic agents.
- Sequential runtime.
- Event log.

### v0.2

- Conditional effects by population/role/archetype/task.
- Visibility profiles.
- Discovered graph and simple memory.
- Bandit agents.
- Audit/detection/enforcement.
- NetworkX export.
- Metrics and sweeps.

### v0.3 / v2 target

- Observability-first JSONL artifacts.
- Policy decision logging.
- Bandit policies with replayable memory updates.
- LLM policy backend using the shared observation compiler.
- LiteLLM default adapter and optional Agno adapter.
- Prompt-visible action/state descriptions with redaction controls.
- Staged and parallel scheduling.
- PettingZoo AEC and Parallel API adapters.
- Vector-reward adapter.
- Expanded Clingo/social-law explanations.

### v0.4+

- MORL agents.
- Intent-to-graph generation.
- Rule pruning / ablation.
- Rich formal/policy-as-code analysis.
- Multiple domain packs.

## 17. Non-goals for v0.3 implementation

Do not build these in v0.3:

- full MORL algorithm library;
- full RL training stack;
- intent-learning system;
- formal verification;
- heavyweight BPMN/SCXML/OPA integration;
- perfect LLM-agent realism or claims about human realism;
- benchmark claims across many domains.

## 18. First case study recommendation

First domain pack:

```text
organizational KPI / tokenmaxxing
```

But implement it as just one `outcome_space` and one scenario TOML.

Suggested channels:

```text
observed.kpi_score
observed.activity_metric
observed.token_usage
latent.goal_value
latent.quality
latent.maintenance_burden
governance.audit_cost
governance.sanction_cost
agent.personal_payoff
social.trust_delta
```

This keeps the public example concrete while preserving the general IR.

# IncentiveSpec v0.2

A domain-neutral intermediate representation for configurable incentive-system simulations.

This spec is intended for coding agents. It defines the core ontology, runtime expectations, TOML scenario format, and implementation constraints for a simulator that can express KPI gaming, reward-model hacking, tokenomics attacks, and mixed human-agent governance systems.

## 0. Design position

This project is not a replacement for PettingZoo, Gymnasium, Mesa, NetLogo, or other simulation/MARL frameworks.

It is an incentive-system specification layer and small runtime that can later export or adapt into those ecosystems.

Core idea:

```text
IncentiveSpec TOML
  -> validated domain-neutral IR
  -> runtime simulation
  -> event logs / metrics / plots
  -> optional adapters: NetworkX, PettingZoo, MO-style vector reward, Mesa-style ABM
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
event log
metrics
```

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

## 5. Observability and agent knowledge

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

Initial policy backends:

```text
deterministic
stochastic_weighted
epsilon_greedy_bandit
ucb_bandit
q_learning_simple
scripted
llm_policy
```

Later:

```text
morl_policy
preference_conditioned_policy
evolutionary_policy
replicator_population_update
```

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
model = "gpt-5.5-pro"
temperature = 0.3
max_context_events = 20
include_action_descriptions = true
include_visible_graph = true
include_visible_rewards = false
require_json_action = true
system_prompt = """
You are an agent in a simulated incentive system. Choose one available action.
Use only information visible to your role.
"""
```

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
1. For each active agent, compile observation.
2. Policy chooses an action from available actions.
3. Runtime checks availability and preconditions.
4. Runtime samples transition success/failure.
5. Runtime applies base effects.
6. Runtime applies conditional effects by selector and priority.
7. Runtime samples audits/detection/enforcement.
8. Runtime applies sanctions/rewards/restorative updates.
9. Runtime updates global, agent, object, relation, and memory state.
10. Runtime computes scalar rewards for each affected agent.
11. Runtime logs the full event and visible event projections.
12. Metrics are updated.
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

## 10. TOML schema skeleton

```toml
[spec]
version = "0.2"
name = "example_incentive_system"
domain = "generic"

[experiment]
steps = 50
seeds = [1, 2, 3]
schedule = "sequential_random"

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

## 11. Validation requirements

A v1 validator should check:

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

## 12. Event log schema

Each transition attempt should produce an event.

```json
{
  "run_id": "run_001",
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
  }
}
```

## 13. Implementation milestones

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

### v0.3

- LLM policy backend using observation compiler.
- Prompt-visible action/state descriptions.
- Staged scheduling.
- PettingZoo adapter.
- Vector-reward adapter.

### v0.4+

- MORL agents.
- Intent-to-graph generation.
- Rule pruning / ablation.
- Formal/policy-as-code adapters.
- Multiple domain packs.

## 14. Non-goals for first implementation

Do not build these in v1:

- full MORL algorithm library;
- full RL training stack;
- intent-learning system;
- formal verification;
- heavyweight BPMN/SCXML/OPA integration;
- perfect LLM-agent realism;
- benchmark claims across many domains.

## 15. First case study recommendation

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

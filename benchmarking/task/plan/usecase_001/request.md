# Usecase 001 — Original Request

> **Captured:** 2026-06-06 11:09 PDT
> **Author:** tony (Tony Chen <tchen7@atlassian.com>)
> **Task type:** plan-quality evaluation (compare and score three candidate plans)
> **Target repo:** `/Users/tchen7/MyProjects/atlassian-agi`
> **Three plans being compared:**
> - Plan A: `/Users/tchen7/.claude/plans/take-a-look-into-fluffy-wand.md`
> - Plan B: `/Users/tchen7/MyProjects/atlassian-agi/data/src/_docs/_plan/00_PLAN_data_builder.md` (the v1.0 backup, see notes in repos.yaml)
> - Plan C: `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/_runtime/tasks/task/task_20260606_012220_87b5ffb3/children/propose/outputs/final_deliverables/output.md`

---

## Verbatim user prompt (the request that produced the three plans)

```
Take a look into /Users/tchen7/MyProjects/atlassian-agi/data/opportunity-studies/tony, we should develop an data-builder package under /Users/tchen7/MyProjects/atlassian-agi/data/src,

We need to build systematic approach to pull project data , layer by layer
(1) - how to get all past projects and all ongoing projects, i.e. how to get all projects
(2) - how to judge project health , first coming with rule based methods? We cannot run llm on all projects at once right? So maybe som rule based methods could filter and rank projects to look into
(3) - we can prioritize identifying projects meeting the following scenario
```
If frontier labs like OpenAI, Anthropic, and Google DeepMind keep compounding (they will), they’re not stopping at “write better code” or “have PR agent” or “build vuln management agent to scan for open vulnerabilities and remediate”.
What frontier labs are now doing is using agencies like Mercor and domain experts to basically go into enterprises and create datasets (let’s call them execution traces) that capture the spirit of what a reasonably competent person in that domain would do in given circumstances (in an environment).
Execution traces: actions people or agents tend to take in a given environment
Environment: the configuration of the problem/domain (the rules of the game)
They then take this and use it to construct evals. Evals (model company shorthand for evaluations) are the critical part of this. We take these execution traces/environments and arrange them in such a way that the distribution mimics the behavior we want to see. Then we can push the model towards this distribution using reinforcement learning.
Early models trained on web-scale data (public code, docs, forums, etc… web data that we all know and love). This is classic next token prediction (NTP). This gave rise to language (hence “language models”) and predictions from patterns. Ostensibly, it gave models “fluency”.
The next step is expert traces + RL. This step evolves over time but the end result is mastery. That’s how we’ve gone from autocomplete to credible coding agents so quickly. Building evals for coding agents is trivial compared to building them for general knowledge work, but the work starts with collecting these execution traces. This is why Anthropic is striking deals with financial institutions to collect information on how real people use Excel.
Web training = pattern mimicry
Expert traces + RL = competence or “understanding”
Another way to think of it, NTP is rote learning, expert traces + RL is apprenticeships / internships / real world environments.
The next target for this is the software lifecycle. The overall path looks like
NTP … coding agents … software agents … white-collar-work agents … PhD/lab agents … ?
If the frontier labs point that same machinery from the above at the entire software lifecycle (planning, decomposition, coordination, review, deploy, incident response) the coordination brain eventually lives above workflow tools like Atlassian, no matter what we build.
And if they run this over the full software lifecycle, they’ll learn things like coordination policy or how to run a good project or how to tell if a sprint is about to go off the rails or how to tell if we are about to have an incident or how to identify org growth challenges. Think: simulating how work gets done, simulating how all enterprise workflows and systems behave and work, how enterprises coordinate and build software/systems.
If that happens before we build our own domain intelligence layer, we compress into storage over time. Essentially, Atlassian becomes a database for frontier intelligence and the value almost entirely accrues to the labs. Not overnight, but steadily.
Examples
So now with all that out of the way… let’s give some vertical application examples (which I think are much less interesting long-term but also very likely easier short-term to build) as well as a few horizontal ones (which is where things get super interesting… think modern AI operating systems for enterprises, etc.)
1. “A project is always green until it’s suddenly red” (Vertical)
Customer Problem
CTOs find out too late that sprints are slipping
What We Build
A live “delivery health score” that predicts probably of sprint failure, under/over scoped epics/tickets etc, project/dependency fragility, PR review risk etc
How We Train It (Without Customer Data)
Extract structural sprint traces… probably things like….
Initial project/epic scope size, project decomposition, story point distribution, mid-sprint scope change/churn/update, PR iteration counts/churn, incident linkages, etc.
Label outcomes like: on-time vs. delayed, incident-light vs. incident-heavy, over budget vs. on-budget, etc.
Train model to map structure → outcome probability, then run RL in synthetic sprint simulators to learn:
	“What decomposition shape minimizes overrun risk?”
Why Us and not Frontier Labs?
This is a context problem, not a content problem. RAG can surface what's in a ticket. We're learning from how tickets behave relative to each other over time. We see multi-year telemetry tied to real outcomes across various systems, they can’t do this (yet) but are absolutely trying to collect this both internally and externally
2. Incident predictor aka "something bad is likely to happen” (Vertical)
Customer Problem
Incidents feel unpredictable, operating orgs and software at scale is mostly flying blind into a storm every day
What We Build
Build models that can predict or identify possible failures, silos/roadblocks/bottlenecks, points of failure and even possible remediations before they happen (incident notebook ready to go if it all does go to hell anyway)
How We Train It
Extract incident graph traces from all the data we already have in Jira, Confluence, Loom, BitBucket, etc.
Cluster graph shapes and then learn which topologies/paths/structures correlate with recurrence/drift/positive outcomes/negative outcomes
Simulate alternative ownership splits, reward reduced resolution time
Why We Can Do It
We see structured incident graphs internally… watching how teams navigate incidents through their tools, not reading what they wrote afterward.
Why They Can’t
Public postmortems don’t expose internal workflows, can’t really piece this information together unless you have access to all the systems
3. Project Critic (or basically any type of business critic) (Vertical)
Customer Problem
Epics planning is messy, mis-scoped, random
What We Build
A project evaluator that can do things like assign risk, make recommendations, plan, organize, marshall resources, flag for higher level of review in parts where needed
Why We Can Do It
We’ve seen millions of project decomposition patterns tied to outcomes across various systems
Why They Can’t
There is no public dataset of multi-year issue trees with outcome labels. This is the equivalent of needing to watch every mathematician work for years with a camera strapped to their head. You can't synthesize it, and you can't scrape it. You have to be the system it happens inside
4. “Swing project meme” (horizontal)


Customer Problem
Teams build something slightly different than intended
What We Build
A system that detects drift between:
Requirement -> Task -> Code -> Delivery -> Incident
How We Train It
Lifecycle traces….requirements node, linked issues/tickets/PRs, deployments/rollbacks/PR updates, incident tags
Learn embeddings for structural alignment patterns and reward alignment patterns that historically led to things like low rework or low incident density
Why We Can Do It
Holy grail of software. We can see almost all of this
Why They Can’t
They see fragments… and most everything they have access to is just a fragment
5. Org Coordination Optimizer aka “work” (horizontal)
Customer Problem
This is the essence of work. How do enterprises scope work, scale it, project plan, budget, etc.
What We Build
A coordination graph model predicting/optimizing/organizing dependencies, process/project/team fragility, delay risk, scope risk, budget risk, etc.
How We Train It
We’ve got the work graph in team nodes, issues, PRs, etc. and we can see things like ownership changes/movement, escalations, re-prioritizations and know when those have issues or delays or cause incidents or meetings to be spun up and the resulting (re)work/drift from those meetings
Simulate structural variations and then reward lower predicted delay probability
Why We Can Do It
We see cross-team coordination patterns at scale
Why They Can’t
They don’t have multi-org private workflow telemetry, and learning this from a handful of orgs is too noisy because all orgs are substantially different

Frontier labs will get here via traces and partnerships IMO. We have a window and a way to build a durable advantage.
```

(4) You also use your imagination and innovation to think about other health issues a project could have. So identify projects of other meaningful health issues.

(5) You probably also want to identify successful major projects as comparison. Thos challenging projects, similar projects but success? Those projects similar to those projects with health problems but succeeded instead? So we can have failed/successful projects for pairwise comparison, do you understand my intent?

(6) after identifying meaningful projects, reconstruct project traces and timeline, and create index of all project artifacts similar but not limited to like /Users/tchen7/MyProjects/atlassian_packages/_plan/atlassian_data_moat_vision/opportunity_studies/project_case_studies/01_GORDIAN_delivery_health , you double check again those project profile , make in-depth and comprehensive enhancement as much as possible , so that we know. Again we need to pull comprehensive and indpeth data from rich sources to have a full picturea of the proejcts.



So to achieve above, fully relying on AI agents is not scalable for collecting data,  say if I have 100 projects etc. let AI agent launch deep research for each of them is time consuming, maybe we can build data builder to have predefined logic to systematically pull comprehensive and in-depth data first, and then ask LLM to further cook on top?

So create such data builder under /Users/tchen7/MyProjects/atlassian-agi/data/src, and then launch agents to significantly expand and enhance /Users/tchen7/MyProjects/atlassian_packages/_plan/atlassian_data_moat_vision/opportunity_studies/project_case_studies? If we want to later use these project profile to post train an LLM for enterprise intelligence, how many projects do you think needed? Your decision and best judgment.




You create a plan first.

I might be wrong, therefore YOU MUST make carefully, thoroughly double check with critical-thinking and honest assesment, make really deep, thorough and accurate investigation; ultrathink. Fulfil my ask properly and elegantly, no ad-hoc, no hacky.
Please make carefully, thoroughly double check with critical-thinking, with really deep, thorough and accurate investigation; ultrathink. Spawn as many agents as possible, do as many iterations as needed, and work on user's ask end to end. DO NOT stop until you get your job done.
```

---

## Follow-up evaluation prompt (the request that this benchmark captures)

The user followed up with this evaluation request (the one being benchmarked):

> Here are a few plans, they might have been updated so re-read the plans
>
> - `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/_runtime/tasks/task/task_20260606_012220_87b5ffb3/children/propose/outputs/final_deliverables/output.md`
> - `/Users/tchen7/MyProjects/atlassian-agi/data/src/_docs/_plan/00_PLAN_data_builder.md`
> - `/Users/tchen7/.claude/plans/take-a-look-into-fluffy-wand.md`
>
> Carefully compare the quality of three plans, identify any issues or problems from each. So compare from various perspectives, and scoring them, plan depth, comprehensiveness, correctness, elegance (design quality) is of top weighted perspective.
>
> I might be wrong, therefore YOU MUST make carefully, thoroughly double check with critical-thinking and honest assesment, make really deep, thorough and accurate investigation; ultrathink. Fulfil my ask properly and elegantly, no ad-hoc, no hacky.
>
> Please make carefully, thoroughly double check with critical-thinking, with really deep, thorough and accurate investigation; ultrathink. Spawn as many agents as possible, do as many iterations as needed, and work on user's ask end to end. DO NOT stop until you get your job done assesing all plans.

---

## Top-weighted evaluation axes (per the request)

1. Plan depth
2. Comprehensiveness
3. Correctness
4. Elegance (design quality)

Operationalizability is captured as a secondary axis but the four above carry top weight.

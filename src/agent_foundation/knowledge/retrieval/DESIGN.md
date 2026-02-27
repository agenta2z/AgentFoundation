# MRS RankEvolve Knowledge Service Design

## The Problem

In scientific modeling and machine learning workflows, the most valuable knowledge often lives in people's heads -- accumulated over years of trial and error. An engineer knows that "initializing LES from a converged RANS solution saves 40% spin-up time," or that "CFL above 0.5 causes divergence for this class of transient problems." This expertise is typically shared in conversations, emails, or run notes, then forgotten or buried.

When building an agent that helps improve models (e.g., a co-science agent for simulation optimization), we need a way to:

1. **Capture** this expertise from free-form text -- the way humans naturally express it
2. **Organize** it automatically -- without asking experts to fill in forms or tag their input
3. **Retrieve** the right knowledge at the right time -- when the agent is working on a relevant task

The Knowledge Service solves this by using an LLM to read free-text input and automatically structure it into a queryable knowledge base that the agent draws from during its work.

---

## How It Works: The Big Picture

```
 Engineer shares experience as free text
          |
          v
    LLM reads and structures it
          |
          v
    Knowledge is stored in three layers
    (profiles, searchable pieces, relationships)
          |
          v
    Agent queries knowledge base at runtime
          |
          v
    Relevant knowledge is injected into the agent's prompt
```

The system has two main phases: **ingestion** (turning free text into structured knowledge) and **retrieval** (surfacing the right knowledge when the agent needs it).

---

## Ingestion: From Free Text to Organized Knowledge

### What the expert provides

Just plain text. No forms, no dropdowns, no required fields. For example:

> *"When running transient LES for jet noise cases, I've found that initializing from a converged RANS solution cuts the spin-up time by roughly 40%. Use a CFL of 0.3 for the first 1000 time steps, then ramp to 0.8. The Smagorinsky constant of 0.1 works better than the default 0.17 for our mesh resolution. Last month we tried dynamic Smagorinsky on case JN-042 and saw 15% better spectral agreement with experimental data."*

### What the system does with it

An LLM reads the text guided by a **structuring prompt** -- a carefully designed instruction set that tells the LLM how to break down and classify the input. The LLM produces a structured JSON output that gets validated and loaded into the knowledge base.

If the LLM output fails validation (malformed JSON, missing fields), the system retries automatically (up to a configurable limit).

The structuring prompt instructs the LLM to:

- **Break the text into discrete pieces** of knowledge, each representing one idea, one fact, one procedure, or one observation
- **Classify each piece** along two dimensions (explained below)
- **Identify entities** mentioned in the text (people, techniques, tools, simulation cases) and capture their attributes as structured profiles
- **Map relationships** between entities (e.g., "technique X was used on case Y", "engineer A specializes in method B")
- **Preserve everything** -- don't summarize or drop information
- **Generate search-friendly keywords** for each piece so it can be found later
- **Exclude sensitive data** like credentials or API keys

### Two dimensions of classification

Each piece of knowledge gets two labels that serve different purposes:

**Knowledge Type** -- groups pieces by *what kind of knowledge they represent*. This is an internal organizational label, useful for filtering and understanding the composition of the knowledge base. The categories are:

- **Fact** -- a factual statement ("Smagorinsky constant of 0.1 works better than default 0.17")
- **Instruction** -- a directive or rule ("Always check mesh quality before CHT runs")
- **Preference** -- a preference or opinion ("Prefer second-order discretization for accuracy")
- **Procedure** -- a multi-step workflow ("Step 1: Initialize with steady-state. Step 2: Switch to transient...")
- **Note** -- a general observation ("Residuals showed unexpected oscillation around iteration 500")
- **Episodic** -- a record of a specific past event ("On case JN-042 we tried dynamic Smagorinsky and saw 15% improvement")

**Info Type** -- controls *where the knowledge gets injected in the agent's prompt*. This is the routing label. When the agent retrieves knowledge, each piece ends up in a specific section of the prompt based on its info type:

- **user_profile** -- goes into the section that tells the agent about the user (who they are, what they specialize in, their preferences)
- **instructions** -- goes into the section that tells the agent how to behave. These are individual, standalone directives: a single rule, a single best practice, a single guardrail. Examples: "Always check mesh quality before CHT runs", "Use second-order discretization for production runs", "Never exceed CFL of 1.0 for explicit time-stepping." Instructions are atomic -- each one stands on its own and the agent follows them independently.
- **context** -- goes into the section that gives the agent background knowledge (facts about techniques, past results, domain knowledge)
- **skills** -- goes into the section that defines reusable, actionable capabilities the agent can invoke. Where instructions are individual rules, a skill is a **coherent, systematic technique** -- it bundles together multiple steps, parameter choices, conditions, and know-how into a named capability. For example, "RANS-to-LES initialization" is a skill that encompasses: start from a converged RANS solution, set initial CFL to 0.3, ramp to 0.8 over 1000 steps, monitor residuals for stability, and switch to production time step. No single instruction captures this -- it's the *orchestrated combination* that forms the skill.

  This is a particularly important info type because the framework **automatically infers skills** from user input. When an engineer describes a technique or workflow (e.g., "I usually initialize LES from a converged RANS, then ramp CFL from 0.3 to 0.8 over 1000 steps"), the system recognizes this as a repeatable skill and extracts it as such -- not just a piece of knowledge to remember, but a concrete capability the agent can apply. The system also **compares new input against existing knowledge** to detect skills that may already be partially captured, merging or refining them rather than creating duplicates. This means the skill library grows organically as experts share their experience, and the agent's repertoire of actionable techniques improves over time without anyone explicitly defining "here is a skill."

This separation matters because the agent treats these prompt sections differently. Instructions are individual rules the agent should *obey*. Context is things the agent should *know about*. User profile is things the agent should *personalize for*. Skills are systematic techniques the agent *can perform* -- coherent, multi-step capabilities it has learned from human expertise. An instruction says "do X." A skill says "here is how to accomplish Y, which involves X along with several other coordinated steps."

Info types are also extensible -- you can define custom prompt sections beyond these defaults if your agent's prompt template calls for it.

---

## Automatic Skill Creation

One of the most important capabilities of the knowledge service is that **skills are not manually defined -- they are automatically synthesized** from user input and existing knowledge.

### How skills emerge from free text

Engineers rarely say "here is a skill." They say things like:

> *"I usually initialize LES from a converged RANS, then ramp CFL from 0.3 to 0.8 over 1000 steps, and keep an eye on residuals before switching to the production time step."*

This is a casual description of a workflow, but the system recognizes it as a **coherent, repeatable technique** and creates a skill entry -- with a name, the full sequence of steps, parameter values, and conditions.

### How skills emerge from scattered knowledge

More importantly, skills don't have to come from a single input. Over time, the knowledge base accumulates **sporadic, related pieces** from different engineers, different sessions, different contexts:

- One engineer shares: *"Always start CHT from a converged flow-only solution"* (stored as an instruction)
- Another mentions: *"For CHT, I set the solid thermal conductivity to match experimental values before coupling"* (stored as a fact)
- A third notes: *"After coupling fluid and solid, reduce the relaxation factor to 0.3 or it oscillates"* (stored as an instruction)
- A run log records: *"Case HT-019: CHT diverged because we forgot to refine the mesh at the interface"* (stored as an episodic note)

Individually, these are just scattered instructions, facts, and notes. But during ingestion, the system **scans existing knowledge** and recognizes that these pieces collectively describe a coherent technique. It then **automatically synthesizes a new skill** -- for example, "Conjugate Heat Transfer Setup" -- that weaves together the initialization strategy, parameter settings, stability precautions, and known pitfalls into a single, structured capability.

### The skill creation process

```
 New user input arrives
          |
          v
 LLM structures it into pieces (facts, instructions, notes, etc.)
          |
          v
 System retrieves existing related knowledge from the store
          |
          v
 LLM examines new pieces + existing pieces together:
   - Do these form a coherent technique that isn't captured as a skill yet?
   - Does this new input extend or refine an existing skill?
   - Are there scattered instructions that, combined, describe a workflow?
          |
          v
 If yes: create a new skill (or update an existing one)
 that synthesizes the related pieces into a named, structured capability
          |
          v
 Skill is stored with info_type = "skills"
 and linked to the source pieces and entities in the graph
```

### Why this matters

Without auto skill creation, the knowledge base would be a flat collection of tips and observations -- useful for search, but the agent would have to figure out on its own how pieces relate and combine. With auto skill creation, the agent receives **pre-assembled techniques** it can apply directly. The difference is:

- **Without skills:** The agent retrieves "start from RANS", "set CFL to 0.3", "ramp to 0.8", "watch residuals" as four separate pieces and has to reason about how they fit together.
- **With skills:** The agent retrieves "RANS-to-LES Initialization" as a single, coherent skill with all steps in order, and can execute it as a known technique.

This also means the knowledge base **gets smarter over time**. Each new piece of input is not just stored -- it's an opportunity to discover new skills or refine existing ones. The skill library grows organically from the accumulation of human experience, without anyone needing to sit down and formally author "skill definitions."

---

## The Three-Layer Knowledge Store

The knowledge base is not a single database. It's three complementary stores, each handling a different shape of information:

```
              +------------------------+
              |     Knowledge Base     |
              |     (orchestrator)     |
              +---+-------+--------+--+
                  |       |        |
         +--------+  +----+---+  +-+----------+
         |           |        |               |
    +----v-----+  +--v-----+  +---v--------+
    | Profiles |  | Pieces |  | Relations  |
    +----------+  +--------+  +------------+
     who/what      searchable   how things
     is this?      knowledge    connect
```

### Profiles (Metadata Store)

Structured key-value records about known entities -- engineers, tools, techniques, simulation cases. These are looked up directly, not searched.

*Example: The system knows that Alice is a CFD engineer on the Aero team who specializes in turbomachinery. When Alice asks the agent a question, her profile is retrieved instantly by key lookup and injected into the user_profile prompt section.*

### Pieces (Knowledge Pieces Store)

The main body of knowledge. Each piece is a chunk of text with its classification labels (knowledge type, info type) and tags. Pieces are retrieved by **search** -- keyword matching, semantic similarity, or both -- so the system finds the most relevant knowledge for the agent's current task.

*Example: When the agent is working on an LES simulation setup, it searches the pieces store and finds the CFL ramp procedure, the Smagorinsky constant insight, and the JN-042 case results -- all ranked by relevance to the current query.*

### Relations (Entity Graph Store)

A graph of how entities connect to each other. Techniques relate to problems, engineers have experience with methods, cases used specific configurations. The graph is traversed (e.g., "starting from this engineer, what techniques have they used, and what problems do those techniques apply to?") to discover relevant knowledge that a text search alone might miss.

Graph edges can link back to specific knowledge pieces, bridging relationship traversal to text content.

*Example: Starting from "LES", the graph reveals it's applicable to jet-noise problems, has a variant called dynamic Smagorinsky, and was used on case JN-042. The edge to JN-042 links to the episodic piece with the 15% improvement result.*

### Why three layers instead of one?

Different questions need different retrieval strategies:

| Question the agent has | Best answered by |
|------------------------|-----------------|
| "Who am I talking to and what do they care about?" | Profile lookup |
| "What do we know about mesh quality for conjugate heat transfer?" | Text search across pieces |
| "What techniques has this engineer used, and what are they good for?" | Graph traversal |

A single query triggers all three layers in parallel. The results are merged and routed into the appropriate prompt sections by info type.

---

## Retrieval: Getting Knowledge to the Agent

When the agent processes a user query, a **Knowledge Provider** sits between the knowledge base and the agent's prompt. It:

1. Sends the query to all three store layers
2. Collects and merges results
3. Groups them by info type
4. Formats each group into text
5. Hands them to the agent as prompt variables

For example, when the agent receives *"Set up an LES simulation for jet noise prediction"*:

- **user_profile section** gets: the engineer's background and specialization
- **instructions section** gets: the RANS initialization procedure, the CFL ramp strategy
- **context section** gets: Smagorinsky constant tuning insights, JN-042 case results

The agent sees relevant expertise, properly organized, without anyone having to manually curate what knowledge applies to this particular query.

---

## Worked Example

### Session 1: First engineer shares experience

**Input (free text):**

> *"When running transient LES for jet noise cases, I've found that initializing from a converged RANS solution cuts the spin-up time by roughly 40%. Use a CFL of 0.3 for the first 1000 time steps, then ramp to 0.8."*

**What the system extracts:**

| What was extracted | Knowledge type | Info type |
|-------------------|---------------|-----------|
| "Initialize transient LES from converged RANS -- cuts spin-up by ~40%" | procedure | instructions |
| "CFL 0.3 for first 1000 steps, then ramp to 0.8" | procedure | instructions |

The system also recognizes these two procedures form a coherent workflow and **automatically creates a skill**: "RANS-to-LES Initialization" -- combining the RANS warm-start with the CFL ramp strategy into a single named technique.

### Session 2: Second engineer adds related knowledge weeks later

**Input (free text):**

> *"The Smagorinsky constant of 0.1 works better than the default 0.17 for our mesh resolution. Also, I always monitor the energy spectrum at the first 500 steps to make sure the LES is resolving enough scales before trusting the results."*

**What happens:**

The system stores these as individual pieces (a fact and an instruction). But it also **retrieves the existing "RANS-to-LES Initialization" skill** and recognizes that the Smagorinsky tuning and the energy spectrum check are directly relevant. It **updates the skill** to incorporate these as additional steps/parameters, producing a richer technique that now covers initialization, CFL ramp, subgrid model tuning, and validation.

### Session 3: A run result comes in

**Input (free text):**

> *"Last month we tried dynamic Smagorinsky on case JN-042 and saw 15% better spectral agreement with experimental data."*

**What happens:**

This is stored as an episodic piece (context). The system links it to the existing skill and techniques in the graph, enriching the knowledge base with evidence of what has worked.

### Later, when the agent is asked to set up a new jet noise LES case:

The agent receives:
- **skills section**: The full "RANS-to-LES Initialization" skill -- a pre-assembled technique covering warm-start, CFL ramp, Smagorinsky tuning, and spectrum validation, synthesized from multiple engineers' inputs over time
- **context section**: The JN-042 case result showing dynamic Smagorinsky improved accuracy by 15%
- **user_profile section**: The engineer's background and specialization

The agent doesn't have to piece together scattered tips. It has a coherent, named technique it can apply directly -- built automatically from the accumulated experience of the team.

---

## Storage Flexibility

All three stores are defined as abstract interfaces, so the backing storage can be swapped without changing the knowledge logic:

| Store | For development / small scale | For production / large scale |
|-------|------------------------------|------------------------------|
| Profiles | In-memory, file, or SQLite | Redis or similar |
| Pieces | In-memory or SQLite full-text search | Vector DB (Chroma, LanceDB) or Elasticsearch |
| Relations | In-memory or file-based graph | Neo4j or similar graph DB |

---

## Key Design Choices

1. **LLM does the organizing** -- Experts just write naturally. The LLM handles all classification, tagging, and structuring. This keeps the barrier to contributing knowledge as low as possible.

2. **Three layers for three access patterns** -- Rather than forcing everything into one store, profiles are looked up, pieces are searched, and relations are traversed. Each layer does what it's best at.

3. **Knowledge type vs. info type** -- Knowledge type is about *what something is* (a fact, a procedure, an observation). Info type is about *where it goes in the agent's prompt* (instructions the agent should follow, context it should reason over, profile info it should personalize with). Separating these lets the same piece be categorized accurately and routed correctly.

4. **Retrieval is query-driven** -- The agent doesn't get the entire knowledge base dumped into its prompt. It gets the pieces most relevant to the current query, found by search and graph traversal, then routed into the right prompt sections.

5. **Backend-agnostic** -- The knowledge logic doesn't care whether pieces live in SQLite or Elasticsearch. Abstract interfaces and thin adapters make it easy to start simple and scale up.

6. **Auditability** -- Every ingestion can be logged with the raw input alongside the structured output, so you can trace back from any piece of stored knowledge to the original text that produced it.

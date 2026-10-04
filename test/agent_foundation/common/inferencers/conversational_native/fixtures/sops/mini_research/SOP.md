# Mini Research

Test fixture SOP for the native conversational orchestrator.

## Phase 0 -- Topic
[__initial__]

[__requires user input__] Ask the user for the research topic with a `clarification` conversation tool. The output variable MUST be named `research_topic`.

## Phase 1 -- Brief
[__depends on__ Phase 0]

Write the brief for `{{ research_topic }}`.

**Tools**[__required__]:
- /write-brief <topic>

## Phase 2 -- Review
[__depends on__ Phase 1]

[__requires user input__] Ask the user to confirm the brief with a `confirmation` conversation tool.

# Native Research

A lightweight research workflow: find out the topic, write a brief, confirm it.

[__keywords__] research brief, native research, write a brief, research sop
[__example_requests__]
- start the native_research SOP
- write me a research brief
- research <topic> for me

## Phase 0 -- Topic
[__initial__]

[__requires user input__] Ask the user what topic they want a research brief on. Use a `clarification` conversation tool. The output variable MUST be named `research_topic`.

## Phase 1 -- Write the brief
[__depends on__ Phase 0]

Write a short research brief on `{{ research_topic }}` by calling the write_brief tool.

**Tools**[__required__]:
- /write-brief <topic>

## Phase 2 -- Confirm
[__depends on__ Phase 1]

[__requires user input__] Show the user the brief and ask them to confirm it looks good, using a `confirmation` conversation tool.

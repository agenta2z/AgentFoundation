# Chief of Staff Agent — Concrete Test Queries

Derived from the product doc: focus areas for the default "Chief of Staff" agent.
Each section maps to a focus area from the doc, with queries ordered from simple → complex.

---

## 1. 📥 Follow-up / What Have I Missed?
*"Prioritize comments on content I created (work items, pages, looms, PRs); replies to my comments; tasks/work items assigned to me; PRs I opened."*

### Triage & Prioritisation
1. `What should I follow up on today? Prioritize: (1) comments on work items/pages/PRs I created, (2) replies to my comments, (3) tasks assigned to me, (4) PRs I opened.`
2. `What Jira tickets assigned to me have been updated or commented on in the last 48 hours that I haven't responded to?`
3. `Which Confluence pages I authored have received new comments in the past week that I haven't replied to?`
4. `Which of my open PRs have reviewer comments I haven't addressed yet?`
5. `Are there any Loom videos I recorded that have received comments or replies I haven't seen?`
6. `What threads am I mentioned in across Jira, Confluence, and PRs that are older than 3 days and still unresolved?`
7. `Show me everything that's waiting on me — grouped by tool (Jira, Confluence, Bitbucket, Loom) and sorted by urgency.`
8. `What decisions or action items from the last 2 weeks did people tag me in that I haven't acknowledged?`

---

## 2. 🧠 Second Brain / Memory
*"Track what you've done, why, and what's next. Maintain context across days/weeks/months."*

### Memory Retrieval
1. `What did I work on last week? Give me a summary grouped by project.`
2. `What decisions did I make in the last month and what was the rationale?`
3. `Last time we discussed the authentication redesign, what was the conclusion?`
4. `What projects have I been most active on in the last 30 days based on my Jira comments, PR activity, and Confluence edits?`
5. `I'm about to jump into the billing refactor. What did I do the last time I touched billing-related work?`
6. `What did I commit to doing in my last 3 weekly team meetings that I haven't done yet?`
7. `What context should I remember before my 2pm meeting about the Q3 roadmap?`
8. `What open questions did I raise in any doc or thread in the past month that were never answered?`

### Memory Consolidation
1. `Summarize what I did this week and save it to my memory so you can reference it later.`
2. `Update my MEMORY.md with the key decisions and outcomes from the last sprint.`
3. `What should be added to my long-term memory from this week's activity that I might need to recall in 3 months?`
4. `Create a memory entry for the architectural decision we made today: we chose Kafka over SQS for the event pipeline because of throughput requirements.`

---

## 3. 📊 Dashboard + Metric Summaries
*"Goals/Projects/Jira/Databricks/Tableau; extract KPIs, deltas, anomalies; link back to charts/issues."*

1. `What's the current status of my team's OKRs? Which goals are on track, at risk, or off track?`
2. `Summarize the Jira epic progress for the platform reliability project. What % is done, what's blocked?`
3. `What are the key metrics for my team's goals this quarter? Any anomalies or concerning trends?`
4. `Which of my tracked Atlas projects have had a status change in the last week?`
5. `Give me a TLDR of the weekly engineering metrics — build pass rate, deploy frequency, incident count, MTTR.`
6. `Which Jira tickets in my current sprint are at risk of not being completed by the end of the sprint?`
7. `Are there any goals owned by my team that haven't been updated in more than 2 weeks?`
8. `Pull the current project health for all initiatives I'm an owner or contributor on and flag anything red.`

---

## 4. ✍️ Writing & Drafting (Personalized)
*"Tone/format adapts by channel (Slack vs Confluence) and persona (IC vs EM)."*

### Slack Drafts
1. `Draft a Slack message to my team letting them know the deploy is delayed by 2 days due to a P1 in the payment service.`
2. `Someone asked me in Slack: "When will the API rate limiting feature ship?" Draft a reply based on the current Jira status.`
3. `Draft a casual Slack update for #eng-platform saying we've resolved the latency spike and sharing what the root cause was.`
4. `Write a short Slack message asking the design team for a review of the new onboarding flow mockups.`

### Confluence Drafts
1. `Create a Confluence page summarizing today's architecture review meeting. Key decisions: we're adopting feature flags for all new rollouts, and deprecating the legacy auth service by Q4.`
2. `Draft a Confluence retrospective page for the Q2 platform reliability project. Include what went well, what didn't, and what we'd do differently.`
3. `Write a project brief for the new notification service we're building. Include: problem statement, proposed solution, success metrics, and open questions.`
4. `Turn these rough bullet points into a polished Confluence design doc: [bullet points]`

### Goal / Status Updates
1. `Write my weekly Atlas goal update for the "Reduce P95 API latency by 20%" goal. Current status: we've achieved 12% reduction, still working on the database query optimization.`
2. `Draft a monthly status update for my manager about what my team has shipped, what's in progress, and what's at risk.`
3. `Write a project update for stakeholders explaining why we're pushing the launch date from May 1 to May 15.`

---

## 5. ⏰ Daily Brief / Morning Briefing
*"Morning briefing: what's urgent, what's due, what changed overnight."*

1. `Give me my morning brief for today. What's urgent, what do I have in my calendar, what changed overnight in my projects?`
2. `What should I prioritize first thing this morning? I have 2 hours before my first meeting.`
3. `What happened while I was offline yesterday? Any fires, decisions made without me, or things that need my response?`
4. `It's Monday morning. What didn't get done last week that's now overdue, and what are the most important things to tackle this week?`
5. `Give me a 5-bullet executive summary of what my team shipped last week and what we're focused on this week.`
6. `I have a 1:1 with my manager in 30 minutes. What should I mention? What's notable from this week?`
7. `What are the top 3 things I absolutely must do today to avoid blocking other people?`

---

## 6. 📋 Task List Generation
*"Generate a reliable task list. Mark tasks completed if done in third-party systems."*

1. `Generate my task list for today based on: Jira tickets assigned to me, PR reviews requested, and Confluence pages I've been asked to review.`
2. `What Jira tickets are assigned to me and due this week that I haven't started yet?`
3. `Create a prioritized to-do list for this sprint based on my Jira board, pending PR reviews, and any threads where I'm the blocker.`
4. `Which tasks from my list last week are now marked done in Jira?`
5. `Are there tasks I said I'd do in Slack or Confluence comments that haven't been tracked in Jira yet? Create tickets for them.`
6. `Add a Jira ticket for: "Migrate legacy auth endpoints to OAuth 2.0" in the Platform project, assign to me, due end of sprint.`
7. `Which of my in-progress Jira tickets haven't had any activity in more than 5 days?`
8. `Scan my recent Slack messages and extract any commitments I made that should become tasks.`

---

## 7. 🔀 PR & Code Workflow
*"Monitor PRs through pipeline → review → merge. React to comments and failures."*

1. `What's the current status of all my open PRs? Which ones are blocked, which need action from me?`
2. `Which of my PRs have had review comments added since I last looked?`
3. `Are any of my PRs failing CI? What are the failures?`
4. `Which PRs am I listed as a reviewer on that I haven't reviewed yet?`
5. `My PR #1234 just got approved. Are there any dependent PRs I should merge first?`
6. `Which feature flags are now safe to clean up based on recent deployment history?`
7. `Alert me if any of my open PRs haven't had activity in more than 3 days.`
8. `What PRs authored by my team are waiting for more than 2 days for a review? Who should I ping?`

---

## 8. 📡 Monitoring & Alerting
*"Always-on watchers: Sentry, Splunk, Slack mentions, stalled tasks, releases."*

1. `Are there any high-priority Sentry errors that spiked in the last 24 hours in services my team owns?`
2. `Did any of the releases deployed yesterday cause an increase in error rate or latency?`
3. `What Slack messages mention my team's services or components from the last 6 hours?`
4. `Are there any blocked Jira tickets in my team's sprint that have been stuck for more than 48 hours?`
5. `Watch the #incidents channel and alert me if anything severity P1 or P2 is posted.`
6. `Is there anything in the Splunk dashboard for the auth service that looks anomalous compared to last week?`
7. `What on-call alerts fired overnight for services in my squad's domain?`

---

## 9. 🔗 Joining the Dots / Cross-Tool Context
*"Connect Slack → Confluence → Loom → chat. Recognize if you responded via Slack to a Confluence comment."*

1. `Someone left a comment on my Confluence design doc about the caching strategy. Did I respond to it anywhere — in Slack, in the doc, in a Jira comment?`
2. `I got feedback on my PR from Alex last Tuesday. Did I address it in a subsequent commit, or is it still open?`
3. `The Q2 reliability initiative — pull together everything related: the Confluence pages, the Jira epic, any Slack threads mentioning it, and the latest Atlas status.`
4. `Is there anything people flagged in Loom comments on my recorded walkthroughs that I haven't followed up on?`
5. `I remember discussing a database migration approach in a meeting 2 weeks ago. Can you find the Loom recording or Confluence notes, and tell me what we decided?`
6. `Show me the full timeline of events for the P1 incident last Thursday: who said what in Slack, what Jira tickets were created, what was deployed.`
7. `Someone asked me in Slack about the status of X. Is the answer already in a Confluence page or Jira ticket I can point them to?`
8. `I think I already responded to this Jira comment in a Slack DM. Can you check and confirm so I don't reply twice?`
9. `Connect all the feedback on the new onboarding flow: Confluence comments, Jira subtasks, Slack threads in #product-feedback, and any Loom walkthroughs. Give me a consolidated view.`
10. `A customer raised an issue that sounds familiar. Have we seen this before? Check Jira history, Confluence pages, and past Slack threads.`

---

## 10. 🧬 Soul / Identity / Persona
*"Soul file that embodies the principles of a Chief of Staff. Identity file for behavior."*

1. `Who are you and what's your role as my Chief of Staff?`
2. `What are your core principles when helping me prioritize work?`
3. `How do you decide what's urgent vs. important?`
4. `What would a great Chief of Staff do differently from a generic assistant?`
5. `How do you handle situations where I'm overcommitted and need to push back on requests?`
6. `What's your approach to helping me manage up (communicating with my manager) vs. managing down (communicating with my team)?`

---

## 11. 🔄 Recurring Ops / Scheduled Automation
*"Cron-style: status reports, metrics, backlog grooming, Atlas updates."*

1. `It's Friday. Generate my weekly team status report summarizing what we shipped, what's in progress, and any blockers for next week.`
2. `Do a backlog grooming pass: flag any Jira tickets that are more than 3 months old, unassigned, or have no activity in 6 weeks.`
3. `Update all my Atlas project statuses based on the current Jira epic progress.`
4. `Generate a monthly digest of my team's contributions: PRs merged, Jira tickets closed, Confluence pages created.`
5. `Every Monday, remind me of: overdue items, items due this week, and any unresolved follow-ups from last week.`
6. `It's end of quarter. Help me draft my team's OKR achievement summary for the all-hands.`

---

## Query Complexity Tiers (for testing)

| Tier | Description | Example queries |
|---|---|---|
| **Tier 1 — Single source, single question** | One tool, one lookup | "What Jira tickets are assigned to me?" |
| **Tier 2 — Single source, filtered/ranked** | One tool, non-trivial filtering | "Which of my PRs have unaddressed comments?" |
| **Tier 3 — Multi-source aggregation** | 2-3 tools, synthesized answer | "Morning brief: calendar + Jira + Slack mentions" |
| **Tier 4 — Cross-tool reasoning** | Tools + memory + time reasoning | "Did I respond to the Confluence comment in Slack?" |
| **Tier 5 — Proactive + agentic** | Agent acts without being asked, loops | "Watch #incidents and alert me on P1s" |

---

## Notes on Testing

- **Start with Tier 1-2** to validate tool connectivity (TWG, Atlassian Search)
- **Tier 3** is where the "magic" moment is — the briefing queries are the best demo
- **Tier 4-5** require memory + cross-tool correlation — hardest to get right, highest value
- The `twg` skill is already configured in the pod with TWG token — queries that go through TWG (Jira, Confluence, Bitbucket, Loom) should work once the plugin ownership issue is resolved
- The `atlassian-search` and `slack-rts` plugins are blocked (uid mismatch) — fixing ownership unlocks Tier 3+ queries that need Slack

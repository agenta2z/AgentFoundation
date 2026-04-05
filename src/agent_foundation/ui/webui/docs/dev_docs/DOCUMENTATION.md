# CoScience Chatbot Demo (React + FastAPI)

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [Core Components](#core-components)
  - [Backend (FastAPI)](#backend-fastapi)
  - [Frontend (React)](#frontend-react)
  - [Experiment Engine](#experiment-engine)
  - [Flow Configuration](#flow-configuration)
- [Key Features](#key-features)
- [Animation System](#animation-system)
- [API Endpoints](#api-endpoints)
- [Data Flow](#data-flow)
- [Running the Demo](#running-the-demo)
- [Development Guide](#development-guide)
- [Additional Guides](#additional-guides)

---

## Overview

The **CoScience Chatbot Demo** (also known as **RankEvolve**) is a sophisticated interactive chatbot application that demonstrates AI-assisted workflows for improving machine learning ranking models. It features:

- **React-based frontend** with Material-UI for a modern chat interface
- **FastAPI backend** for REST API and WebSocket support
- **Experiment Flow Engine** for scripted, multi-step conversation demos
- **Real-time progress animations** with phase-based timing
- **File viewer** for displaying research documents and code documentation

The primary use case is demonstrating a multi-agent deep research workflow where multiple AI agents (OpenAI, Gemini, Claude, MetaMate) work in parallel to research and propose improvements to ranking models.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User (Browser)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         React Frontend (Port 3000 dev)                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  App.js                                                              │    │
│  │  - Chat Message Display                                             │    │
│  │  - Progress Animation Rendering                                     │    │
│  │  - Collapsible Progress Sections                                    │    │
│  │  - File Viewer (Drawer Panel)                                       │    │
│  │  - Suggested Action Buttons                                         │    │
│  │  - HTTP Polling (200ms) for Animation State                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │ HTTP/REST
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend (Port 8087 prod)                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Routes                                                              │    │
│  │  - /api/chat/send          → Process user message                   │    │
│  │  - /api/chat/progress      → Get animation state (polling)          │    │
│  │  - /api/chat/action        → Handle suggested action clicks         │    │
│  │  - /api/experiment/files/* → Serve flow files                       │    │
│  │  - /ws/*                   → WebSocket routes                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ExperimentService                                                   │    │
│  │  - State management                                                 │    │
│  │  - Time-based animation calculations                                │    │
│  │  - Step advancement logic                                           │    │
│  │  - Progress phase transitions                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ExperimentFlowEngine                                                │    │
│  │  - Loads flow.json configuration                                    │    │
│  │  - Manages step navigation                                          │    │
│  │  - Reads experiment files                                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Flow Configuration (flow.json)                       │
│  - Steps with messages, delays, progress sections                           │
│  - Suggested actions (continue, inferencer, input_prefix)                   │
│  - Pre/Post messages with animations                                        │
│  - Progress headers with user input fields                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
chatbot_demo_react/
├── BUCK                          # Buck2 build targets
├── run_demo.sh                   # Main launch script (production)
├── run_backend.py                # Backend entry point for buck2
├── run_frontend.sh               # Frontend dev server script
├── run_coscience.sh              # Alternative launch script
├── experiment_engine.py          # Core experiment flow engine
│
├── backend/                      # FastAPI backend
│   ├── __init__.py
│   ├── main.py                   # FastAPI app initialization
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── chat_routes.py        # Chat API endpoints
│   │   ├── experiment_routes.py  # Experiment/file endpoints
│   │   └── websocket_routes.py   # WebSocket handlers
│   └── services/
│       ├── __init__.py
│       └── experiment_service.py # Business logic & state
│
├── frontend/                     # Production React build output
│   ├── index.html
│   ├── logo.png
│   ├── asset-manifest.json
│   └── static/
│       ├── css/
│       └── js/
│
├── react/                        # React source code
│   ├── package.json
│   ├── yarn.lock
│   ├── public/
│   │   ├── index.html
│   │   └── logo.png
│   ├── src/
│   │   ├── App.js                # Main React component
│   │   ├── App.css               # Styles
│   │   ├── index.js              # Entry point
│   │   └── setupProxy.js         # Dev proxy config
│   └── build/                    # Built assets
│
└── experiment_configs/           # Flow configurations
    └── coscience_experiment/
        ├── flow.json             # Main flow definition
        └── files/
            ├── prompts/          # Prompt templates
            │   ├── codebase_documentation_prompt.txt
            │   ├── codebase_investigation_prompt.txt
            │   ├── deepresearch_external_prompt.txt
            │   ├── deepresearch_internal_prompt.txt
            │   └── proposal_prompt_openai.txt
            └── context/
                ├── codebase_documentation/  # Sphinx docs
                │   ├── _build/html/        # Generated HTML
                │   ├── conf.py
                │   ├── index.rst
                │   ├── architecture.rst
                │   └── ...
                ├── proposals/              # Generated proposals
                │   ├── merged_proposal.md
                │   ├── proposal_chatgpt.md
                │   ├── proposal_claude.md
                │   ├── proposal_gemini.md
                │   └── proposal_metamate.md
                └── researches/             # Research results
                    ├── external_research_chatgpt.md
                    ├── external_research_claude.md
                    ├── external_research_gemini.md
                    └── internal_research_metamate.md
```

---

## Core Components

### Backend (FastAPI)

#### `backend/main.py`

The main FastAPI application entry point that:
- Creates the FastAPI app with CORS middleware
- Registers API routers for chat, experiment, and WebSocket routes
- Initializes the `ExperimentService` with the default flow
- Serves the React frontend in production mode
- Provides health check and SPA fallback routing

**Key initialization:**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    experiment_service = ExperimentService(flow_name="coscience_experiment")
    yield
```

#### `backend/routes/chat_routes.py`

Handles all chat-related API endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat/send` | POST | Send user message, process step, return state |
| `/api/chat/messages` | GET | Get all messages in current session |
| `/api/chat/action` | POST | Handle suggested action button clicks |
| `/api/chat/progress` | GET | Get current animation state (polling) |
| `/api/chat/progress/continue` | POST | Continue from progress_header phase |
| `/api/chat/reset` | POST | Reset experiment to beginning |

#### `backend/routes/experiment_routes.py`

Handles experiment status and file serving:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/experiment/status` | GET | Get current experiment state |
| `/api/experiment/progress` | GET | Polling fallback for progress |
| `/api/experiment/complete-step` | POST | Complete animation, get step messages |
| `/api/experiment/files/{path}` | GET | Get file content for viewer |
| `/api/experiment/static-html/{path}` | GET | Serve HTML docs with correct MIME types |
| `/api/experiment/flow-info` | GET | Get flow metadata |

#### `backend/services/experiment_service.py`

The core business logic service that:
- Wraps `ExperimentFlowEngine` for FastAPI
- Manages animation state with **time-based calculations**
- Tracks messages, progress sections, and suggested actions
- Handles step advancement and auto-advance logic
- Provides stateless progress state calculations

**Key classes:**
- `ChatMessage`: Represents a chat message (role, content, file_path, message_type)
- `ProgressState`: Animation state with phase, revealed counts, timing info
- `ExperimentState`: Full experiment state for frontend
- `ExperimentService`: Main service class

### Frontend (React)

#### `react/src/App.js`

A comprehensive React component (~2300 lines) that manages:

**State Management:**
- `messages`: Chat message history
- `completedSteps`: Completed steps with user_message + post_messages + progress_sections
- `pendingUserMessage`: User message triggering current animation
- `progressState`: Current animation state from backend
- `suggestedActions`: Action buttons to display
- `collapsedSections`: Track which progress sections are collapsed
- `visibleSections`: Track which sections are visible based on appearance_delay

**Animation Phases:**
1. `pre_delay`: Wait before showing pre_messages
2. `pre_messages`: Show messages sequentially (replacing)
3. `progress_header`: Show header with input, wait for continue
4. `progress`: Show progress sections animation
5. `post_delay`: Wait after progress completes
6. `post_messages`: Show messages sequentially (appending)
7. `complete`: Animation finished

**Key Features:**
- HTTP polling every 200ms for animation updates
- Auto-scroll to bottom on new messages
- Collapsible progress sections with expand/collapse
- File viewer drawer for documents
- Markdown rendering with syntax highlighting
- Debug logging accessible via `window.debugLogs`

**Dependencies:**
- Material-UI (`@mui/material`, `@mui/icons-material`)
- `react-markdown` for markdown rendering
- `react-syntax-highlighter` for code blocks

### Experiment Engine

#### `experiment_engine.py`

Defines the data structures and logic for experiment flows:

**Data Classes:**
- `InputFieldConfig`: User input field configuration
- `ContinueButtonConfig`: Continue button styling
- `ProgressHeaderConfig`: Header shown before progress sections
- `MessageConfig`: Single message with delays and file references
- `ParallelGroup`: Messages to display in parallel
- `ProgressMessage`: A message within a progress section
- `ProgressSection`: Collapsible section with animated messages
- `SuggestedAction`: Action button configuration
- `InferencerActionConfig`: Configuration for AI inferencer integration
- `StepConfig`: Full step configuration
- `ExperimentFlowConfig`: Complete flow configuration

**Key Classes:**
- `ExperimentFlowEngine`: State machine for flow playback
- `ExperimentFlowLoader`: Loads and validates flow.json files

### Flow Configuration

#### `experiment_configs/coscience_experiment/flow.json`

Defines the complete demo workflow with 6 steps:

| Step | ID | Description |
|------|-----|-------------|
| 0 | `step_0` | Welcome message with workflow plan |
| 1 | `step_1` | Codebase investigation & documentation |
| 2 | `step_2` | Parallel deep research (4 AI agents) |
| 3 | `step_3` | Parallel proposal generation |
| 4 | `step_4` | Merge proposals into unified plan |
| 5 | `step_5` | Completion & next iteration prompt |

**Flow Structure:**
```json
{
  "metadata": {
    "name": "CoScience Demo",
    "description": "...",
    "code_entry_point": "fbcode/minimal_viable_ai/models/main_feed_mtml/model_roo_v0.py",
    "context_files_root": "fbcode/_tony_dev/...",
    "group_sections_by_step": true,
    "default_inferencer": { "type": "devmate", "config": {...} }
  },
  "steps": [
    {
      "id": "step_1",
      "role": "assistant",
      "pre_delay": {"min": 2.0, "max": 3.0},
      "post_delay": {"min": 0.8, "max": 1.5},
      "keep_progress_sections": true,
      "pre_messages": [...],
      "progress_sections": [...],
      "progress_header": {...},
      "post_messages": [...],
      "wait_for_user": true,
      "suggested_actions": {...}
    }
  ]
}
```

---

## Key Features

### 1. Phase-Based Animation System

The animation system uses **time-based calculations** rather than stateful tracking:

```
Timeline:
├── pre_delay (2-3s)          → Wait before starting
├── pre_messages (varies)     → Show "🤔 Thinking..." messages
├── progress_header (wait)    → User input prompt (if configured)
├── progress (varies)         → Animate progress sections in parallel
├── post_delay (0.8-1.5s)     → Wait after progress
├── post_messages (varies)    → Show result messages
└── complete                  → Animation finished
```

### 2. Progress Sections

Multiple progress sections can animate in **parallel** with:
- **Staggered appearance**: `appearance_delay` controls when section appears
- **Message delays**: Each message has its own reveal timing
- **Collapsibility**: Sections can be expanded/collapsed
- **Prompt links**: Each section can link to its prompt file

Example section:
```json
{
  "slot": "openai_research",
  "title": "🔍 OpenAI Deep Research",
  "prompt_file": "files/prompts/deepresearch_external_prompt.txt",
  "collapsible": true,
  "message_delay_multiplier": 3.2,
  "messages": [
    {"content": "Scanning RecSys '24, KDD '24...", "delay": 1.5},
    {"content": "Analyzing 'Embedding Collapse' theory...", "delay": 2.0}
  ]
}
```

### 3. Suggested Actions

Actions that users can take after each step:

| Action Type | Description |
|-------------|-------------|
| `continue` | Advance to next step |
| `input_prefix` | Start user input with prefix |
| `inferencer` | Trigger AI inferencer (DevMate) |
| `expand_input` | Expand a collapsed input field |

### 4. Input Fields

User input can be embedded in messages or progress headers:
- **In post_messages**: Collect user feedback or customization
- **In progress_header**: Allow query customization before research
- **Collapsible**: Can be initially collapsed with "Click to customize" button

### 5. File Viewer

Side panel drawer for viewing:
- Markdown files (rendered)
- HTML documentation (iframe with proper MIME types)
- Research results and proposals
- Prompt templates

### 6. Auto-Advance

Steps can be configured with `wait_for_user: false` to automatically advance:
- Backend captures completed step data
- Frontend processes completion before showing next animation
- Enables seamless multi-step sequences

---

## Animation System

### Time-Based State Calculation

The backend calculates animation state based on elapsed time:

```python
def get_progress_state(self) -> ProgressState:
    elapsed_ms = current_time_ms - start_time_ms

    # Calculate which phase we're in
    if elapsed_ms < pre_delay_end:
        phase = "pre_delay"
    elif elapsed_ms < pre_messages_end:
        phase = "pre_messages"
        # Calculate which pre_message to show
    elif waiting_for_progress_header:
        phase = "progress_header"
    elif elapsed_ms < progress_end:
        phase = "progress"
        # Calculate revealed_counts per section
    # ... etc
```

### Frontend Polling

The frontend polls `/api/chat/progress` every 200ms:

```javascript
pollingIntervalRef.current = setInterval(async () => {
  const response = await fetch(`${API_BASE}/chat/progress`);
  const progressData = await response.json();

  // Update state based on phase
  setProgressState(progressData);
  setAnimationPhase(progressData.phase);

  // Handle completion
  if (progressData.phase === 'complete') {
    // Save to completedSteps
    // Clear animation state
    // Stop polling
  }
}, 200);
```

### Section Visibility

Sections appear based on `appearance_delay_ms`:

```javascript
useEffect(() => {
  if (progressState?.phase === 'progress') {
    const elapsed = Date.now() - progressStartTimeClient;
    progressState.sections.forEach(section => {
      const isVisible = elapsed >= section.appearance_delay_ms;
      newVisible[section.slot] = isVisible;
    });
    setVisibleSections(newVisible);
  }
}, [progressStartTimeClient, progressState]);
```

---

## API Endpoints

### Chat Routes

#### POST `/api/chat/send`
Send user message and process experiment step.

**Request:**
```json
{ "message": "Start the RankEvolve workflow" }
```

**Response:**
```json
{
  "current_step_index": 1,
  "current_step_id": "step_1",
  "is_complete": false,
  "messages": [...],
  "suggested_actions": { "message": "...", "actions": [...] },
  "is_waiting_for_user": true,
  "is_animating": true,
  "progress_sections": [...],
  "revealed_counts": {"section1": 3, "section2": 5}
}
```

#### GET `/api/chat/progress`
Get current animation state for polling.

**Response:**
```json
{
  "sections": [...],
  "revealed_counts": {...},
  "is_animating": true,
  "is_complete": false,
  "current_step_id": "step_1",
  "phase": "progress",
  "pre_messages": [...],
  "post_messages": [...],
  "current_pre_message_index": 1,
  "current_post_message_count": 0,
  "keep_progress_sections": true,
  "completed_steps": [...],
  "progress_header": {...},
  "waiting_for_progress_header": false
}
```

#### POST `/api/chat/action`
Handle suggested action button click.

**Request:**
```json
{ "index": 0 }
```

#### POST `/api/chat/progress/continue`
Continue from progress_header phase.

**Request:**
```json
{
  "user_input": {
    "deep_research_query": "What are the latest advancements..."
  }
}
```

### Experiment Routes

#### GET `/api/experiment/files/{path}`
Get file content for viewer.

#### GET `/api/experiment/static-html/{path}`
Serve HTML files with proper MIME types (for Sphinx docs).

#### GET `/api/experiment/flow-info`
Get flow metadata.

---

## Data Flow

### User Message Flow

```
1. User types message → sendMessage()
2. Set pendingUserMessage (for UI rendering)
3. POST /api/chat/send
4. Backend: process_user_input() → _process_current_step()
5. Backend: start_progress_animation() if step has progress
6. Response: { is_animating: true, ... }
7. Frontend: startProgressPolling()
8. Poll /api/chat/progress every 200ms
9. Update progressState, render animations
10. On complete: save to completedSteps, clear animation
```

### Animation State Flow

```
Backend (time-based):                Frontend (polling):
┌─────────────────┐                  ┌─────────────────┐
│ start_time_ms   │                  │ poll every 200ms│
│ cumulative_     │───────────────▶│ GET /progress   │
│   delays_ms     │                  │                 │
│                 │                  │ Update state:   │
│ get_progress_   │                  │ - phase         │
│   state():      │                  │ - revealed_     │
│   elapsed =     │                  │   counts        │
│   now - start   │                  │ - pre/post msgs │
│   calculate     │                  │                 │
│   phase +       │                  │ Render:         │
│   revealed      │                  │ - pre_message   │
│   counts        │                  │ - sections      │
└─────────────────┘                  │ - post_messages │
                                     └─────────────────┘
```

### Step Completion Flow

```
1. Animation reaches phase='complete'
2. Frontend detects completion in polling
3. If keep_progress_sections=true:
   - Save step to completedSteps with:
     - step_id
     - user_message (from pendingUserMessage)
     - post_messages
     - progress_sections (with revealed_count)
4. Auto-collapse completed sections
5. Clear progressState
6. Show suggested_actions
7. Stop polling
8. Wait for user action
```

---

## Running the Demo

### Production Mode

```bash
# Navigate to the project directory
cd /path/to/chatbot_demo_react

# Run the unified launch script
./run_demo.sh

# With debug logging
./run_demo.sh --debug

# Custom port
PORT=8088 ./run_demo.sh
```

The script:
1. Sets Meta proxy for yarn
2. Builds React frontend (`yarn build`)
3. Copies build to `frontend/` directory
4. Launches FastAPI via buck2 with SSL support

**Access:** `https://{hostname}:8087`

### Development Mode

```bash
# Terminal 1: Backend
./run_demo.sh --dev

# Terminal 2: Frontend (in react/ directory)
cd react
yarn start
```

**Frontend dev server:** `http://localhost:3000` (proxies to backend)

### Buck2 Commands

```bash
# Build and run backend
buck2 run //_tony_dev/ScienceModelingTools/tools/ui/chatbot_demo_react:run_backend -- --port 8087 --host "::"

# With SSL
buck2 run //_tony_dev/ScienceModelingTools/tools/ui/chatbot_demo_react:run_backend -- \
  --port 8087 --host "::" \
  --ssl-keyfile /etc/pki/tls/certs/${HOSTNAME}.key \
  --ssl-certfile /etc/pki/tls/certs/${HOSTNAME}.crt
```

---

## Development Guide

### Adding a New Step

1. Edit `experiment_configs/coscience_experiment/flow.json`
2. Add a new step object with:
   - `id`: Unique step identifier
   - `role`: "assistant" or "user"
   - `pre_messages`: Messages shown before progress
   - `progress_sections`: Animated progress sections
   - `post_messages`: Messages shown after progress
   - `suggested_actions`: Action buttons
   - `wait_for_user`: Whether to wait for user action

### Adding a New Progress Section

```json
{
  "slot": "my_section",
  "title": "🔍 My Section Title",
  "prompt_file": "files/prompts/my_prompt.txt",
  "collapsible": true,
  "initial_state": "expanded",
  "message_delay_multiplier": 1.5,
  "appearance_delay": {"min": 2.0, "max": 5.0},
  "messages": [
    {"content": "Starting analysis...", "delay": 1.0},
    {"content": "Processing data...", "delay": 1.5},
    {"content": "✅ Complete!", "delay": 1.0}
  ]
}
```

### Adding Input Fields

**In post_messages:**
```json
{
  "content": "Please provide your feedback:",
  "input_field": {
    "variable_name": "user_feedback",
    "placeholder": "Enter feedback...",
    "multiline": true,
    "optional": true,
    "collapsible": true,
    "initially_collapsed": true
  }
}
```

**In progress_header:**
```json
"progress_header": {
  "content": "## Configure Your Query\n\nCustomize the research query below:",
  "input_field": {
    "variable_name": "custom_query",
    "default_value": "What are the latest advancements...",
    "multiline": true
  },
  "continue_button": {
    "label": "🚀 Start Research",
    "style": "primary"
  }
}
```

### Debugging

**Browser Console:**
```javascript
// View all debug logs
window.debugLogs

// Copy to clipboard
window.copyDebugLogs()

// Filter by type
window.filterDebugLogs('completion')
window.filterDebugLogs(['state', 'polling'])

// Get available log types
window.getDebugLogTypes()

// Show current state
window.showState()
```

**Backend Logging:**
```bash
# Enable debug mode
./run_demo.sh --debug
```

### Frontend Development

**Key hooks and state:**
- `useCallback` for stable function references
- `useRef` for values accessed in closures (e.g., `completionHandledRef`)
- `useEffect` for side effects (polling, visibility updates)

**Important patterns:**
- Refs mirror state for use in async callbacks (avoid stale closures)
- `completionHandledRef.current` checked before state updates
- `baseMessageCountRef.current` for correct message indexing

---

## Building Sphinx Documentation

The experiment configs include RST (reStructuredText) documentation that gets compiled to HTML using Sphinx. This section covers how to build and troubleshoot the documentation.

### Directory Structure

Each experiment config can have its own codebase documentation:

```
experiment_configs/
├── coscience_experiment/
│   └── files/context/codebase_documentation/
│       ├── _build/html/          # Generated HTML (gitignored)
│       ├── conf.py               # Sphinx configuration
│       ├── index.rst             # Main documentation index
│       ├── architecture/         # Architecture docs
│       ├── datasets/             # Dataset docs
│       └── TARGETS               # Buck build target for Sphinx
│
└── coscience_experiment_public/
    └── files/context/codebase_documentation/
        ├── _build/html/          # Generated HTML
        ├── conf.py
        ├── index.rst
        └── ...
```

### Building Documentation with Buck2

The recommended way to build Sphinx documentation at Meta is using Buck2 with the existing Sphinx target:

```bash
# Navigate to the documentation source directory
cd experiment_configs/coscience_experiment_public/files/context/codebase_documentation

# Build using the Buck2 Sphinx target from coscience_experiment
buck2 run //_tony_dev/ScienceModelingTools/tools/ui/chatbot_demo_react/experiment_configs/coscience_experiment/files/context/codebase_documentation:sphinx -- -M html . _build/html
```

This command:
1. Runs the Sphinx build target defined in the TARGETS file
2. `-M html` specifies HTML output format
3. `.` is the source directory (current directory with RST files)
4. `_build/html` is the output directory

### TARGETS File for Sphinx

The Sphinx target is defined in `codebase_documentation/TARGETS`:

```python
load("@fbcode_macros//build_defs:native_rules.bzl", "buck_filegroup")
load("@fbcode_macros//build_defs:python_binary.bzl", "python_binary")

oncall("ads_ranking_for_feed")

python_binary(
    name = "sphinx",
    main_module = "sphinx.cmd.build",
    par_style = "xar",
    deps = [
        "fbsource//third-party/pypi/sphinx:sphinx",
        "fbsource//third-party/pypi/sphinx-rtd-theme:sphinx-rtd-theme",
    ],
)

buck_filegroup(
    name = "source",
    srcs = glob([
        "**/*.rst",
        "**/*.py",
    ]),
)
```

### Alternative: Using fbSphinx (Meta's Sphinx Integration)

For more advanced documentation needs, you can use Meta's fbSphinx:

```bash
# Install fbSphinx
sudo feature install fbsphinx

# Build documentation
fbsphinx build -e //path/to/your/docs:my-docs

# Preview documentation
fbsphinx preview -e //path/to/your/docs:my-docs
```

### Common Issues and Solutions

#### Issue: "sphinx-build: command not found"

**Cause**: The `sphinx-build` command is not directly available on devservers.

**Solution**: Use Buck2 to run Sphinx instead:

```bash
buck2 run //_tony_dev/.../codebase_documentation:sphinx -- -M html . _build/html
```

#### Issue: "File not found" error in File Viewer

**Cause**: The file path in `flow.json` is incorrect or points outside the experiment's base directory.

**Background**: The backend has a security check in `experiment_routes.py` (lines 254-262) that prevents serving files outside the experiment's base directory. Paths using `../` to reference parent directories will be blocked.

**Solution**:
1. Ensure the HTML documentation is built within the experiment's directory
2. Update `flow.json` to use a path within the experiment directory:

```json
// Bad: Path goes outside experiment directory (blocked by security check)
"file": "../coscience_experiment/files/context/codebase_documentation/_build/html/index.html"

// Good: Path within experiment directory
"file": "files/context/codebase_documentation/_build/html/index.html"
```

#### Issue: Intersphinx warnings about unreachable inventories

**Cause**: The devserver may not have direct internet access to fetch external documentation inventories.

**Solution**: These warnings can be safely ignored as they don't affect the local documentation build. The warnings look like:

```
WARNING: failed to reach any of the inventories with the following issues:
intersphinx inventory 'https://docs.python.org/3/objects.inv' not fetchable...
```

### Workflow: Adding Documentation to a New Experiment Config

1. **Copy the documentation structure** from an existing experiment:
   ```bash
   cp -r coscience_experiment/files/context/codebase_documentation \
         my_new_experiment/files/context/codebase_documentation
   ```

2. **Update the RST files** with your content

3. **Build the HTML documentation**:
   ```bash
   cd my_new_experiment/files/context/codebase_documentation
   buck2 run //_tony_dev/.../coscience_experiment/files/context/codebase_documentation:sphinx -- -M html . _build/html
   ```

4. **Reference the documentation in flow.json**:
   ```json
   {
     "content": "✅ Here is the detailed documentation",
     "type": "text",
     "file": "files/context/codebase_documentation/_build/html/index.html"
   }
   ```

### Serving HTML Documentation in the File Viewer

The backend serves HTML documentation through the `/api/experiment/static-html/{path}` endpoint, which:
- Sets proper MIME types for HTML, CSS, JS, and other assets
- Allows Sphinx-generated HTML to load its static assets correctly
- Performs security checks to ensure files are within the experiment directory

---

## Summary

The CoScience Chatbot Demo is a sophisticated demonstration application showcasing:

1. **Modern Web Stack**: React + FastAPI with Material-UI
2. **Time-Based Animations**: Stateless calculations for smooth progress animations
3. **Flexible Flow System**: JSON-configurable experiment flows
4. **Multi-Agent Simulation**: Parallel progress sections for AI agents
5. **Interactive Elements**: User input fields, suggested actions, file viewer
- **Production Ready**: SSL support, Buck2 builds, Meta proxy configuration

The codebase is well-structured with clear separation between:
- **Frontend presentation** (React components, state management)
- **Backend logic** (FastAPI routes, experiment service)
- **Flow configuration** (JSON definitions, prompt files, context documents)

---

## Additional Guides

- **[System Design Document](../SYSTEM_DESIGN.md)** - Comprehensive system design documentation for RankEvolve, including the Evolve methodology, system architecture, research cycle workflow, SYNAPSE architecture, and future roadmap.
- **[NPM Installation Guide](NPM_INSTALLATION_GUIDE.md)** - How to install npm/yarn packages on Meta devservers, including troubleshooting common errors with registry access and proxy configuration.

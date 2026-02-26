.. _infrastructure-build-deploy:

==================
Build & Deployment
==================

Build System
============

TypeScript Build
----------------

OpenClaw uses **tsdown** (esbuild-based bundler) configured in
``tsdown.config.ts``:

- Entry points: ``src/index.ts``, ``src/entry.ts``, ``src/plugin-sdk/``,
  hooks, and other modules
- Output: ``dist/`` directory
- Format: ESM

.. code-block:: bash

   pnpm build          # Build TypeScript + type-check
   pnpm tsgo           # Type-check only (native TS checker)

UI Build
--------

The Control UI uses Vite for building Lit web components:

.. code-block:: bash

   # Built as part of the main build
   cd ui && vite build

Native Apps
-----------

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Platform
     - Build Tool
     - Notes
   * - macOS
     - Swift Package Manager
     - ``apps/macos/Package.swift``; package via ``scripts/package-mac-app.sh``
   * - iOS
     - XcodeGen + xcodebuild
     - ``apps/ios/``
   * - Android
     - Gradle
     - ``apps/android/app/build.gradle.kts``

Deployment Options
==================

Docker
------

Production Dockerfile at the repo root:

.. code-block:: bash

   docker build -t openclaw .
   docker run -p 18789:18789 -v openclaw-data:/data openclaw

Docker Compose (``docker-compose.yml``) provides gateway + CLI services.

Additional Dockerfiles:

- ``Dockerfile.sandbox`` — Sandbox container
- ``Dockerfile.sandbox-browser`` — Browser sandbox
- ``Dockerfile.sandbox-common`` — Common sandbox base

Fly.io
------

Configured via ``fly.toml``:

- App: ``openclaw``
- Region: IAD
- Resources: 2 shared vCPUs, 2GB RAM
- Storage: Persistent volume at ``/data``

.. code-block:: bash

   fly deploy

Render
------

Configured via ``render.yaml``:

- Docker web service
- Starter plan

npm
---

Published as the ``openclaw`` package:

.. code-block:: bash

   npm install -g openclaw
   openclaw setup

Self-hosted
-----------

Direct Node.js installation:

.. code-block:: bash

   npx openclaw setup
   openclaw gateway run

Package Manager
===============

- **Primary**: pnpm 10.23.0
- **Also supported**: Bun (for TypeScript execution in dev)
- Lockfile: ``pnpm-lock.yaml``
- Patches: ``patches/`` directory with ``pnpm.patchedDependencies``

CI/CD
=====

GitHub Actions workflows in ``.github/workflows/``:

- ``ci.yml`` — Lint, format, typecheck, build, test (Linux/macOS/Windows)
- ``docker-release.yml`` — Docker image builds
- ``install-smoke.yml`` — Installation smoke tests
- ``sandbox-common-smoke.yml`` — Sandbox image tests

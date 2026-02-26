.. _infrastructure-testing:

=======
Testing
=======

OpenClaw uses Vitest v4 as its testing framework with V8 coverage.

Test Configuration
==================

Multiple Vitest configs serve different test scenarios:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Config File
     - Command
     - Purpose
   * - ``vitest.config.ts``
     - ``pnpm test``
     - Main unit tests
   * - ``vitest.unit.config.ts``
     - ``pnpm test:fast``
     - Fast unit tests
   * - ``vitest.e2e.config.ts``
     - ``pnpm test:e2e``
     - End-to-end tests
   * - ``vitest.live.config.ts``
     - ``pnpm test:live``
     - Live tests with real API keys
   * - ``vitest.gateway.config.ts``
     - (gateway tests)
     - Gateway-specific tests
   * - ``vitest.extensions.config.ts``
     - ``pnpm test:extensions``
     - Extension plugin tests
   * - ``ui/vitest.config.ts``
     - ``pnpm test:ui``
     - UI component tests (browser mode)

Test Naming Conventions
=======================

- Unit tests: ``*.test.ts`` (colocated with source)
- E2E tests: ``*.e2e.test.ts``
- Live tests: ``*.live.test.ts``

Coverage Thresholds
===================

V8 coverage with these minimum thresholds:

- **Lines**: 70%
- **Functions**: 70%
- **Statements**: 70%
- **Branches**: 55%

Run coverage: ``pnpm test:coverage``

Docker Tests
============

Integration tests that run in Docker containers:

- ``pnpm test:docker:live-models`` — Live model tests
- ``pnpm test:docker:live-gateway`` — Live gateway tests
- ``pnpm test:docker:onboard`` — Onboarding E2E tests

Source: ``vitest.config.ts``, ``test/``

.. _index:

=======================================
OpenClaw Technical Documentation
=======================================

OpenClaw is a self-hosted, multi-channel AI gateway that connects 20+ messaging
platforms to configurable AI agents powered by multiple LLM providers. It
orchestrates agent workflows, manages persistent memory, and extends agent
capabilities through a markdown-based skills system.

This documentation provides both **machine-readable technical specifications**
and **human-friendly tutorials** for developers working with or contributing to
the OpenClaw codebase.

.. note::

   This documentation covers the internal architecture and implementation
   details. For user-facing guides, see the `Mintlify docs <https://docs.openclaw.ai>`_.

Getting Started
===============

- New to OpenClaw? Start with :doc:`architecture/index` for a high-level overview.
- Want to understand how messages flow? Read :doc:`tutorials/tracing-a-message`.
- Interested in the agent system? Dive into :doc:`orchestration/index`.
- Working with memory/knowledge? See :doc:`knowledge/index`.
- Building skills? Check :doc:`skills/index` and :doc:`tutorials/adding-a-skill`.

.. toctree::
   :maxdepth: 3
   :caption: Contents

   architecture/index
   orchestration/index
   knowledge/index
   skills/index
   channels/index
   gateway/index
   plugins/index
   infrastructure/index
   tutorials/index
   glossary

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

.. _channels-index:

===============
Channel System
===============

OpenClaw connects to 20+ messaging platforms through a unified channel
abstraction layer. Channels are implemented either as core modules in ``src/``
or as extension plugins in ``extensions/``.

Architecture
============

.. code-block:: text

   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │   WhatsApp   │  │   Telegram   │  │   Discord    │  ...
   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
          │                 │                 │
   ┌──────▼─────────────────▼─────────────────▼───────┐
   │              Channel Adapter Layer                │
   │          (ChannelMessagingAdapter)                │
   └──────────────────────┬───────────────────────────┘
                          │
   ┌──────────────────────▼───────────────────────────┐
   │                   Gateway                        │
   │            Message Routing Engine                 │
   └──────────────────────────────────────────────────┘

Each channel implements the ``ChannelMessagingAdapter`` interface, which
provides a uniform API for sending/receiving messages, managing conversations,
and handling platform-specific features (reactions, threads, media, etc.).

Core vs Extension Channels
==========================

**Core channels** are built into the OpenClaw source code:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Channel
     - Source
     - Library
   * - WhatsApp
     - ``src/web/`` (WhatsApp Web)
     - ``@whiskeysockets/baileys``
   * - Telegram
     - ``src/telegram/``
     - ``grammy``
   * - Discord
     - ``src/discord/``
     - ``@buape/carbon``
   * - Slack
     - ``src/slack/``
     - ``@slack/bolt``
   * - Signal
     - ``src/signal/``
     - Custom integration
   * - iMessage
     - ``src/imessage/``
     - BlueBubbles API
   * - IRC
     - ``src/irc/`` (if present)
     - Custom
   * - Web Chat
     - ``src/webchat/``
     - Built-in
   * - LINE
     - (via config)
     - ``@line/bot-sdk``
   * - Lark/Feishu
     - (via config)
     - ``@larksuiteoapi/node-sdk``
   * - Google Chat
     - ``src/googlechat/``
     - Google APIs

**Extension channels** are plugins in ``extensions/``:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Channel
     - Extension
     - Description
   * - MS Teams
     - ``extensions/msteams/``
     - Microsoft Teams via Bot Framework
   * - Matrix
     - ``extensions/matrix/``
     - Matrix protocol
   * - Zalo OA
     - ``extensions/zalo/``
     - Zalo Official Account
   * - Zalo User
     - ``extensions/zalouser/``
     - Zalo User messaging
   * - Tlon
     - ``extensions/tlon/``
     - Tlon/Urbit protocol

.. toctree::
   :maxdepth: 2

   plugin-sdk
   channel-list

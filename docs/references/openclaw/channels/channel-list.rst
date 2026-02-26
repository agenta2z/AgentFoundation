.. _channels-channel-list:

============
Channel List
============

Complete list of all supported messaging channels with their implementation
type, library, and key features.

.. list-table::
   :header-rows: 1
   :widths: 15 10 15 15 45

   * - Channel
     - Type
     - Library
     - Source
     - Notes
   * - WhatsApp
     - Core
     - baileys
     - ``src/web/``
     - WhatsApp Web protocol; QR code pairing; group support; media; reactions
   * - Telegram
     - Core
     - grammy
     - ``src/telegram/``
     - Bot API; webhooks; inline keyboards; markdown formatting; media
   * - Discord
     - Core/Ext
     - carbon
     - ``src/discord/``, ``extensions/discord/``
     - Bot + OAuth; guild management; threads; components; slash commands
   * - Slack
     - Core
     - @slack/bolt
     - ``src/slack/``
     - Bot + OAuth; slash commands; blocks; threading; streaming
   * - Signal
     - Core
     - Custom
     - ``src/signal/``
     - Signal protocol integration; group support
   * - iMessage
     - Core
     - BlueBubbles
     - ``src/imessage/``
     - Via BlueBubbles server; macOS required
   * - Web Chat
     - Core
     - Built-in
     - ``src/webchat/``
     - Browser-based chat widget
   * - LINE
     - Core
     - @line/bot-sdk
     - Config-based
     - LINE Messaging API
   * - Lark/Feishu
     - Core
     - @larksuiteoapi/node-sdk
     - Config-based
     - Lark/Feishu bot integration
   * - Google Chat
     - Core
     - Google APIs
     - ``src/googlechat/``
     - Google Workspace integration
   * - MS Teams
     - Extension
     - Bot Framework
     - ``extensions/msteams/``
     - Microsoft Teams bot
   * - Matrix
     - Extension
     - matrix-js-sdk
     - ``extensions/matrix/``
     - Matrix protocol; E2EE support
   * - Zalo OA
     - Extension
     - Custom
     - ``extensions/zalo/``
     - Zalo Official Account API
   * - Zalo User
     - Extension
     - Custom
     - ``extensions/zalouser/``
     - Zalo user messaging
   * - Tlon
     - Extension
     - Custom
     - ``extensions/tlon/``
     - Tlon/Urbit protocol
   * - IRC
     - Core
     - Custom
     - Config-based
     - IRC protocol support

Channel Capabilities
====================

Each channel declares its capabilities, which affect how the agent interacts
with it:

- **Threading**: Whether the channel supports message threads
- **Reactions**: Whether emoji reactions are supported
- **Media**: Types of media supported (images, audio, video, documents)
- **Formatting**: Markdown, HTML, or plain text support
- **Streaming**: Whether partial/streaming responses are supported
- **Slash commands**: Whether the channel supports native slash commands

The agent's system prompt includes channel capability information so the LLM
can adapt its responses accordingly (e.g., using markdown on Discord but
plain text on SMS).

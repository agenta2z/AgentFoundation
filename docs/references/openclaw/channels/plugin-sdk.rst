.. _channels-plugin-sdk:

==========================
Channel Plugin SDK
==========================

Channel plugins implement a set of typed adapter interfaces exported from
``src/plugin-sdk/index.ts``. These interfaces define how channels send and
receive messages, manage configuration, and handle pairing/authentication.

Core Interfaces
===============

ChannelMessagingAdapter
-----------------------

The primary interface for message exchange:

.. code-block:: typescript

   interface ChannelMessagingAdapter {
     // Send a message to a channel destination
     send(params: {
       to: string;
       text: string;
       media?: MediaAttachment[];
       replyTo?: string;
       threadId?: string;
     }): Promise<SendResult>;

     // Receive and route inbound messages
     onMessage(handler: MessageHandler): void;

     // Channel-specific capabilities
     capabilities(): ChannelCapabilities;
   }

ChannelConfigAdapter
--------------------

Handles channel-specific configuration UI and validation:

.. code-block:: typescript

   interface ChannelConfigAdapter {
     // Configuration fields for the setup wizard
     configFields(): ConfigField[];

     // Validate configuration
     validate(config: Record<string, unknown>): ValidationResult;
   }

ChannelPairingAdapter
---------------------

Manages authentication and connection pairing:

.. code-block:: typescript

   interface ChannelPairingAdapter {
     // Initiate pairing (e.g., QR code for WhatsApp)
     pair(params: PairingParams): Promise<PairingResult>;

     // Check connection status
     status(): ConnectionStatus;

     // Disconnect
     disconnect(): Promise<void>;
   }

Plugin Registration
===================

Channel plugins register themselves via the ``openclaw.plugin.json`` manifest:

.. code-block:: json

   {
     "name": "my-channel",
     "version": "1.0.0",
     "type": "channel",
     "main": "dist/index.js",
     "channel": {
       "id": "my-channel",
       "displayName": "My Channel",
       "messaging": true,
       "pairing": true
     }
   }

The plugin's main entry point exports adapter factory functions that the
gateway calls during channel initialization.

Source: ``src/plugin-sdk/index.ts``, ``src/channels/``

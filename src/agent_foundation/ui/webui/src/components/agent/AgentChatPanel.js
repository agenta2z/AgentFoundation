/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * AgentChatPanel — Main real-mode chat UI container.
 *
 * Uses useAgentChat hook for WebSocket communication.
 * Reuses existing MarkdownRenderer and ChatInput components.
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Box, Container, Paper, Typography } from '@mui/material';
import { useAgentChat } from '../hooks/useAgentChat';
import { MarkdownRenderer } from '../common/MarkdownRenderer';
import { ChatInput } from '../chat/ChatInput';
import { AgentStatusBar } from './AgentStatusBar';
import { StreamingMessage } from './StreamingMessage';
import { CommandAutocomplete } from './CommandAutocomplete';

export function AgentChatPanel() {
  const {
    messages,
    streamingMessage,
    isStreaming,
    config,
    connectionStatus,
    taskPhase,
    sendMessage,
    cancelRequest,
    clearMessages,
    isConnected,
  } = useAgentChat();

  const [inputValue, setInputValue] = useState('');
  const [showAutocomplete, setShowAutocomplete] = useState(false);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingMessage]);

  const handleSubmit = useCallback((e) => {
    e.preventDefault();
    if (!inputValue.trim() || !isConnected) return;

    if (inputValue.trim() === '/clear') {
      sendMessage(inputValue);
      clearMessages();
    } else {
      sendMessage(inputValue);
    }
    setInputValue('');
    setShowAutocomplete(false);
  }, [inputValue, isConnected, sendMessage, clearMessages]);

  const handleInputChange = useCallback((value) => {
    setInputValue(value);
    setShowAutocomplete(value.startsWith('/') && !value.includes(' '));
  }, []);

  const handleAutocompleteSelect = useCallback((command) => {
    setInputValue(command + ' ');
    setShowAutocomplete(false);
  }, []);

  const renderMessage = (msg, index) => {
    const isUser = msg.role === 'user';
    const isSystem = msg.role === 'system';
    const isError = msg.role === 'error';

    let bgColor = 'background.paper';
    let borderColor = 'divider';
    if (isUser) {
      bgColor = 'primary.dark';
      borderColor = 'primary.main';
    } else if (isSystem) {
      bgColor = 'rgba(255, 255, 255, 0.05)';
      borderColor = 'rgba(255, 255, 255, 0.15)';
    } else if (isError) {
      bgColor = 'rgba(244, 67, 54, 0.1)';
      borderColor = 'error.main';
    }

    return (
      <Box
        key={index}
        sx={{
          display: 'flex',
          justifyContent: isUser ? 'flex-end' : 'flex-start',
          mb: 2,
        }}
      >
        <Paper
          elevation={0}
          sx={{
            p: 2,
            maxWidth: isSystem ? '90%' : '80%',
            backgroundColor: bgColor,
            borderRadius: 2,
            border: '1px solid',
            borderColor: borderColor,
          }}
        >
          {isSystem ? (
            <Typography
              variant="body2"
              sx={{
                fontFamily: 'monospace',
                fontSize: '0.85rem',
                color: 'text.secondary',
                whiteSpace: 'pre-wrap',
              }}
            >
              {msg.content}
            </Typography>
          ) : isError ? (
            <Typography variant="body2" sx={{ color: 'error.main' }}>
              ⚠️ {msg.content}
            </Typography>
          ) : (
            <Box sx={{ '& p': { m: 0 }, '& pre': { overflow: 'auto' } }}>
              <MarkdownRenderer content={msg.content} />
            </Box>
          )}
        </Paper>
      </Box>
    );
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <AgentStatusBar
        connectionStatus={connectionStatus}
        model={config.model}
        targetPath={config.target_path}
        taskPhase={taskPhase}
        isStreaming={isStreaming}
      />

      <Container maxWidth="md" sx={{ flex: 1, display: 'flex', flexDirection: 'column', py: 2, overflow: 'hidden' }}>
        <Box sx={{ flex: 1, overflow: 'auto', mb: 2, px: 1 }}>
          {messages.length === 0 && !streamingMessage && (
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', opacity: 0.6 }}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h5" sx={{ mb: 1 }}>🤖 RankEvolve Agent</Typography>
                <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                  Type a message to chat, or use commands like /task, /help, /model
                </Typography>
              </Box>
            </Box>
          )}

          {messages.map((msg, index) => renderMessage(msg, index))}

          {streamingMessage && (
            <StreamingMessage
              content={streamingMessage.content}
              metadata={streamingMessage.metadata}
              taskPhase={taskPhase}
            />
          )}

          {isStreaming && !streamingMessage?.content && (
            <Box sx={{ display: 'flex', justifyContent: 'flex-start', mb: 2 }}>
              <Paper elevation={0} sx={{ p: 2, backgroundColor: 'background.paper', borderRadius: 2, border: '1px solid', borderColor: 'divider' }}>
                <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                  Thinking...
                </Typography>
              </Paper>
            </Box>
          )}

          <div ref={messagesEndRef} />
        </Box>

        <Box sx={{ position: 'relative' }}>
          {showAutocomplete && (
            <CommandAutocomplete
              input={inputValue}
              onSelect={handleAutocompleteSelect}
              onClose={() => setShowAutocomplete(false)}
            />
          )}
          <ChatInput
            value={inputValue}
            onChange={handleInputChange}
            onSubmit={handleSubmit}
            disabled={!isConnected}
          />
        </Box>
      </Container>
    </Box>
  );
}

export default AgentChatPanel;

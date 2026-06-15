/**
 * AgentMessageBubble — foldable AI response bubble with header bar.
 *
 * Features:
 * - Header bar: agent name, "View Prompt" button (if promptData available),
 *   "View Full Response" button (if content exceeds maxHeight), fold toggle arrow
 * - Foldable body: max-height 320px, overflow hidden (content clipped)
 * - ThinkingFold, MarkdownRenderer, ChatWidgetRenderer, SessionContextBar inside body
 */

import React, { useState, useRef, useEffect } from 'react';
import {
  Avatar,
  Box,
  Button,
  Collapse,
  Typography,
} from '@mui/material';
import { SmartToy as SmartToyIcon, CheckCircle as CheckCircleIcon } from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import { MarkdownRenderer } from '../common/MarkdownRenderer';
import { ThinkingFold, stripToolsToInvoke } from './ThinkingFold';
import { SessionContextBar } from './SessionContextBar';
import ChatWidgetRenderer from '../protocol/ChatWidgetRenderer';

const DEFAULT_BODY_HEIGHT = 320;
const DEFAULT_MAX_WIDTH = '75%';

function formatTime(ts) {
  if (!ts) return '';
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export function AgentMessageBubble({ message, onViewPrompt, onViewFullResponse }) {
  const theme = useTheme();
  const maxBodyHeight = theme.custom?.layout?.responseMaxHeight || DEFAULT_BODY_HEIGHT;
  const maxWidth = theme.custom?.layout?.responseMaxWidth || DEFAULT_MAX_WIDTH;
  const [folded, setFolded] = useState(false);
  const [contentOverflows, setContentOverflows] = useState(false);
  const contentRef = useRef(null);

  const agentName = message.agent_name || message.metadata?.agent_name || 'AI Assistant';
  const hasThinking = message.thinkingContent && message.responsePhase !== 'no_tags';
  // Show "View Prompt" if we have inline prompt data OR a turn_number to fetch from disk
  const hasPromptData = Boolean(message.promptData?.rendered_prompt) || Boolean(message.turnNumber);

  // Detect if body content overflows max-height after render
  useEffect(() => {
    const el = contentRef.current;
    if (el) {
      setContentOverflows(el.scrollHeight > maxBodyHeight);
    }
  }, [message.content, message.thinkingContent]);

  return (
    <Box sx={{ display: 'flex', justifyContent: 'flex-start', mb: 2 }}>
      <Avatar
        sx={{ mr: 1, width: 32, height: 32, bgcolor: '#4a90d9', flexShrink: 0, mt: 0.5 }}
      >
        <SmartToyIcon sx={{ fontSize: 18 }} />
      </Avatar>

      <Box sx={{ maxWidth: maxWidth, flex: 1, minWidth: 0 }}>
        {/* ── Header bar ─────────────────────────────────────────── */}
        <Box
          onClick={() => setFolded(f => !f)}
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 0.75,
            px: 1.5,
            py: 0.6,
            cursor: 'pointer',
            backgroundColor: 'rgba(255,255,255,0.04)',
            border: '1px solid rgba(255,255,255,0.08)',
            borderBottom: folded ? '1px solid rgba(255,255,255,0.08)' : 'none',
            borderRadius: folded ? '4px 16px 16px 16px' : '4px 16px 0 0',
            '&:hover': { backgroundColor: 'rgba(255,255,255,0.07)' },
            transition: 'border-radius 0.15s',
          }}
        >
          {/* Agent name */}
          <Typography
            variant="caption"
            sx={{ fontWeight: 700, color: '#4a90d9', flex: 1, userSelect: 'none' }}
          >
            {agentName}
          </Typography>

          {/* View Prompt button */}
          {hasPromptData && (
            <Button
              size="small"
              variant="outlined"
              onClick={(e) => { e.stopPropagation(); onViewPrompt && onViewPrompt(message); }}
              sx={{
                fontSize: '0.62rem',
                height: 20,
                py: 0,
                px: 0.75,
                minWidth: 0,
                borderColor: 'rgba(255,255,255,0.2)',
                color: 'text.secondary',
                '&:hover': { borderColor: 'primary.light', color: 'primary.light' },
              }}
            >
              View Prompt
            </Button>
          )}

          {/* View Full Response button — always shown for viewing thinking + response + raw */}
          {onViewFullResponse && (
            <Button
              size="small"
              variant="outlined"
              onClick={(e) => { e.stopPropagation(); onViewFullResponse && onViewFullResponse(message.rawContent || message.content || ''); }}
              sx={{
                fontSize: '0.62rem',
                height: 20,
                py: 0,
                px: 0.75,
                minWidth: 0,
                borderColor: 'rgba(255,255,255,0.2)',
                color: 'text.secondary',
                '&:hover': { borderColor: 'secondary.light', color: 'secondary.light' },
              }}
            >
              Full Response
            </Button>
          )}

          {/* Complete checkmark + fold toggle */}
          <CheckCircleIcon sx={{ color: 'success.main', fontSize: 16, ml: 0.25 }} />
          <Typography
            sx={{
              color: 'text.disabled',
              fontSize: '0.7rem',
              userSelect: 'none',
              lineHeight: 1,
              ml: 0.25,
            }}
          >
            {folded ? '▸' : '▾'}
          </Typography>
        </Box>

        {/* ── Collapsible body ────────────────────────────────────── */}
        <Collapse in={!folded}>
          <Box
            ref={contentRef}
            sx={{
              backgroundColor: 'rgba(255,255,255,0.05)',
              border: '1px solid rgba(255,255,255,0.08)',
              borderTop: 'none',
              borderBottom: message.sessionContext ? 'none' : '1px solid rgba(255,255,255,0.08)',
              px: 2,
              py: 1.5,
              // Square bottom when sessionContext footer is present below
              // (so the body and the footer visually merge); rounded bottom otherwise.
              borderRadius: message.sessionContext ? '0' : '0 0 16px 16px',
              lineHeight: 1.6,
              fontSize: '0.9rem',
              color: 'text.primary',
              maxHeight: maxBodyHeight,
              overflow: 'auto',
              '& p': { m: 0 },              // top-level paragraphs: no margin (keeps simple replies tight)
              '& p:last-child': { mb: 0 },  // no trailing gap below last paragraph
              '& li p': { m: 0 },           // p inside li: keep tight (li provides spacing)
              '& li': { mb: 0.25 },         // gentle gap between list items
              '& ul, & ol': { pl: 2.5, mt: 0.5, mb: 0.5 },  // indent + breathing room
              '& pre': { overflow: 'auto' },
            }}
          >
            {/* Collapsible thinking section */}
            {hasThinking && (
              <ThinkingFold thinkingContent={message.thinkingContent} />
            )}

            {/* Main response content */}
            <MarkdownRenderer content={message.content} />

            {/* Widgets (approval, choice, etc.) */}
            {message.widgets?.length > 0 && (
              <Box sx={{ mt: 1.5 }}>
                <ChatWidgetRenderer widgets={message.widgets} />
              </Box>
            )}
          </Box>
          {/* Session context bar — sits OUTSIDE the scrollable body so it stays
              pinned at the bottom of the bubble even when the user scrolls a
              long response. (Previously it was inside the overflow container
              and would scroll out of view, making it feel like it was missing
              on long messages.) */}
          {message.sessionContext && (
            <Box
              sx={{
                px: 2,
                py: 1,
                backgroundColor: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderTop: 'none',
                borderRadius: '0 0 16px 16px',
              }}
            >
              <SessionContextBar
                usedLabel={message.sessionContext.usedLabel}
                totalLabel={message.sessionContext.totalLabel}
                percentage={message.sessionContext.percentage}
              />
            </Box>
          )}
        </Collapse>

        {/* Timestamp */}
        <Typography
          variant="caption"
          sx={{ color: 'text.secondary', mt: 0.5, ml: 0.5, display: 'block' }}
        >
          {formatTime(message.timestamp)}
        </Typography>
      </Box>
    </Box>
  );
}

export default AgentMessageBubble;

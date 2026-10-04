/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * AgentStreamSection — ProgressSection-style collapsible per-agent streaming
 * box used by the Implementation tab's RunStreamPanel / StreamingView.
 *
 * Ported from RankEvolve (components/agent/AgentStreamSection.js); the only
 * change is importing `MarkdownRenderer` + `stripToolsToInvoke` from the
 * shared AgentFoundation barrels (the latter lives in `chat/ThinkingFold`).
 */

import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Collapse,
  Chip,
  CircularProgress,
  Button,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  CheckCircle as CheckCircleIcon,
  OpenInNew as ViewAllIcon,
  Code as CodeIcon,
  Psychology as ThinkingIcon,
} from '@mui/icons-material';
import { MarkdownRenderer } from '../../../../common/MarkdownRenderer';
import { stripToolsToInvoke, stripResponseTags } from '../../../../chat/ThinkingFold';

export { stripToolsToInvoke, stripResponseTags };

const AGENT_INFO = {
  base: { icon: '🔵', label: 'Base Agent' },
  review: { icon: '🟣', label: 'Review Agent' },
  system: { icon: '⚙️', label: 'System' },
  welcome: { icon: '👋', label: 'Welcome Message' },
};

function parseAgentId(agentId) {
  const match = String(agentId || '').match(/^(\w+?)(?:_round(\d+))?$/);
  const role = match ? match[1] : agentId;
  const roundNum = match && match[2] ? parseInt(match[2], 10) : null;
  const info = AGENT_INFO[role] || { icon: '🤖', label: role };
  const displayLabel = roundNum
    ? `${info.icon} ${info.label} (Round ${roundNum})`
    : `${info.icon} ${info.label}`;
  return { info, roundNum, displayLabel };
}

/** Blinking cursor shown during active streaming. */
function BlinkingCursor() {
  return (
    <Box
      component="span"
      sx={{
        display: 'inline-block',
        width: 8,
        height: 16,
        backgroundColor: 'primary.main',
        ml: 0.5,
        verticalAlign: 'text-bottom',
        animation: 'blink 1s step-end infinite',
        '@keyframes blink': {
          '0%, 100%': { opacity: 1 },
          '50%': { opacity: 0 },
        },
      }}
    />
  );
}

/** Collapsible "Thinking" subsection shown when <Response> tag is detected. */
function ThinkingFold({ thinkingContent }) {
  const [expanded, setExpanded] = useState(false);

  if (!thinkingContent) return null;

  const charCount = thinkingContent.length;

  return (
    <Box sx={{ mb: 1.5 }}>
      <Box
        onClick={() => setExpanded(!expanded)}
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 0.5,
          cursor: 'pointer',
          py: 0.5,
          px: 1,
          borderRadius: 1,
          backgroundColor: 'rgba(255, 255, 255, 0.03)',
          '&:hover': { backgroundColor: 'rgba(255, 255, 255, 0.06)' },
        }}
      >
        <ThinkingIcon sx={{ fontSize: 14, color: 'text.disabled' }} />
        <Typography
          variant="caption"
          sx={{ color: 'text.disabled', fontWeight: 500, userSelect: 'none' }}
        >
          {expanded ? '▾' : '▸'} Thinking ({charCount.toLocaleString()} chars)
        </Typography>
      </Box>
      <Collapse in={expanded}>
        <Box
          sx={{
            mt: 0.5,
            ml: 1,
            pl: 1.5,
            borderLeft: '2px solid rgba(255, 255, 255, 0.08)',
            opacity: 0.5,
            color: 'text.secondary',
            maxHeight: 200,
            overflow: 'auto',
            '& p': { m: 0 },
          }}
        >
          <MarkdownRenderer content={stripToolsToInvoke(thinkingContent)} />
        </Box>
      </Collapse>
    </Box>
  );
}

export function AgentStreamSection({
  agentId,
  content,
  isComplete,
  isPlaceholder = false,
  onViewAll,
  onViewPrompt,
  turnNumber,
  defaultCollapsed = false,
  showStatus = true,
  fitContent = false,
  thinkingContent,
  responseContent,
  responsePhase,
}) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);
  const { displayLabel } = parseAgentId(agentId);

  const isThinking = responsePhase === 'pre_response';
  const hasResponse = responsePhase === 'in_response' || responsePhase === 'post_response';
  const isResponseStreaming = responsePhase === 'in_response';

  const headerSuffix = isThinking && !isComplete ? ' · Thinking...' : '';

  return (
    <Paper
      elevation={0}
      sx={{
        mb: 1.5,
        backgroundColor: 'rgba(74, 144, 217, 0.08)',
        borderRadius: 2,
        border: '1px solid',
        borderColor: isComplete ? 'success.main' : 'primary.dark',
        overflow: 'hidden',
        opacity: isComplete && collapsed ? 0.85 : 1,
        transition: 'opacity 0.2s',
      }}
    >
      <Box
        onClick={() => setCollapsed(!collapsed)}
        sx={{
          p: 1.5,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          cursor: 'pointer',
          backgroundColor: 'rgba(0, 0, 0, 0.2)',
          borderBottom: collapsed ? 'none' : '1px solid rgba(255,255,255,0.1)',
          '&:hover': { backgroundColor: 'rgba(0, 0, 0, 0.3)' },
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {collapsed ? (
            <ExpandMoreIcon sx={{ color: 'primary.light', fontSize: 20 }} />
          ) : (
            <ExpandLessIcon sx={{ color: 'primary.light', fontSize: 20 }} />
          )}
          <Typography
            variant="subtitle2"
            sx={{ color: 'primary.light', fontWeight: 600 }}
          >
            {displayLabel}{turnNumber ? ` (Turn ${turnNumber})` : ''}
            {headerSuffix && (
              <Typography
                component="span"
                variant="subtitle2"
                sx={{ color: 'text.disabled', fontWeight: 400, ml: 0.5 }}
              >
                {headerSuffix}
              </Typography>
            )}
          </Typography>
          {showStatus && isComplete && (
            <Chip
              label="Complete"
              size="small"
              sx={{
                height: 18,
                fontSize: '0.6rem',
                backgroundColor: 'rgba(76,175,80,0.15)',
                color: '#81c784',
              }}
            />
          )}
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {turnNumber && onViewPrompt && (
            <Button
              size="small"
              startIcon={<CodeIcon sx={{ fontSize: 14 }} />}
              onClick={(e) => {
                e.stopPropagation();
                onViewPrompt(turnNumber);
              }}
              sx={{
                fontSize: '0.7rem',
                textTransform: 'none',
                color: 'text.secondary',
                minWidth: 'auto',
                py: 0.25,
                px: 0.75,
                '&:hover': {
                  color: 'primary.light',
                  backgroundColor: 'rgba(255,255,255,0.1)',
                },
              }}
            >
              View Prompt
            </Button>
          )}
          {content && onViewAll && (
            <Button
              size="small"
              startIcon={<ViewAllIcon sx={{ fontSize: 14 }} />}
              onClick={(e) => {
                e.stopPropagation();
                onViewAll(agentId, content, turnNumber);
              }}
              sx={{
                fontSize: '0.7rem',
                textTransform: 'none',
                color: 'text.secondary',
                minWidth: 'auto',
                py: 0.25,
                px: 0.75,
                '&:hover': {
                  color: 'primary.light',
                  backgroundColor: 'rgba(255,255,255,0.1)',
                },
              }}
            >
              View Full Response
            </Button>
          )}
          {showStatus && (isComplete ? (
            <CheckCircleIcon sx={{ color: 'success.main', fontSize: 18 }} />
          ) : (
            <CircularProgress size={14} thickness={4} />
          ))}
        </Box>
      </Box>

      <Collapse in={!collapsed}>
        <Box
          sx={{
            p: 2,
            ...(fitContent ? {} : { maxHeight: 350 }),
            overflow: 'auto',
            '& p': { m: 0 },
            '& pre': { overflow: 'auto' },
            ...(isPlaceholder && { opacity: 0.6, fontStyle: 'italic' }),
          }}
        >
          {isThinking && (
            <>
              <Box sx={{ opacity: 0.5, color: 'text.secondary' }}>
                <MarkdownRenderer content={stripToolsToInvoke(thinkingContent || '')} />
              </Box>
              {!isComplete && <BlinkingCursor />}
            </>
          )}

          {hasResponse && (
            <>
              <ThinkingFold thinkingContent={thinkingContent} />
              <MarkdownRenderer content={stripToolsToInvoke(responseContent || '')} />
              {isResponseStreaming && !isComplete && <BlinkingCursor />}
            </>
          )}

          {!isThinking && !hasResponse && (
            <>
              <MarkdownRenderer content={stripToolsToInvoke(stripResponseTags(content || ''))} />
              {!isComplete && !isPlaceholder && <BlinkingCursor />}
            </>
          )}
        </Box>
      </Collapse>
    </Paper>
  );
}

export default AgentStreamSection;

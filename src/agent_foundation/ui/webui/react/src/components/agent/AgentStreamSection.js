/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * AgentStreamSection — ProgressSection-style collapsible per-agent streaming box.
 * Features dark title bar with expand/collapse, max-height content area,
 * spinner/checkmark status, "View All" button, and <Response>-tag-aware
 * thinking fold that separates LLM reasoning from the final response.
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
import { useTheme, alpha } from '@mui/material/styles';
import {
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  CheckCircle as CheckCircleIcon,
  OpenInNew as ViewAllIcon,
  Code as CodeIcon,
  Psychology as ThinkingIcon,
} from '@mui/icons-material';
import { MarkdownRenderer } from '../common/MarkdownRenderer';

const AGENT_INFO = {
  base: { icon: '🔵', label: 'Base Agent' },
  review: { icon: '🟣', label: 'Review Agent' },
  system: { icon: '⚙️', label: 'System' },
  welcome: { icon: '👋', label: 'Welcome Message' },
};

function parseAgentId(agentId) {
  const match = agentId.match(/^(\w+?)(?:_round(\d+))?$/);
  const role = match ? match[1] : agentId;
  const roundNum = match && match[2] ? parseInt(match[2]) : null;
  const info = AGENT_INFO[role] || { icon: '🤖', label: role };
  const displayLabel = roundNum
    ? `${info.icon} ${info.label} (Round ${roundNum})`
    : `${info.icon} ${info.label}`;
  return { info, roundNum, displayLabel };
}

/** Strip <Response> and </Response> tags from raw content for clean display. */
export function stripResponseTags(text) {
  if (!text) return text;
  return text
    .replace(/<Response>/g, '')
    .replace(/<\/Response>/g, '')
    .trim();
}

/**
 * Strip ```json ToolsToInvoke ... ``` blocks from content for clean display.
 * The tool invocations are handled by the backend — showing raw JSON to the user
 * adds no value and can cause react-markdown rendering issues.
 */
export function stripToolsToInvoke(text) {
  if (!text) return text;
  return text.replace(/```json\s+ToolsToInvoke\n[\s\S]*?```/g, '').trim();
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
  const theme = useTheme();

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
          backgroundColor: theme.custom.surfaces.cardBg,
          '&:hover': { backgroundColor: theme.custom.surfaces.cardBorder },
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
            borderLeft: `2px solid ${theme.custom.surfaces.overlayMedium}`,
            opacity: 0.5,
            color: 'text.secondary',
            maxHeight: 200,
            overflow: 'auto',
            '& p': { m: 0 },
          }}
        >
          <MarkdownRenderer content={thinkingContent} />
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
  // Response-tag-aware props
  thinkingContent,
  responseContent,
  responsePhase,
}) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);
  const theme = useTheme();
  const { displayLabel } = parseAgentId(agentId);

  // Determine what to display based on responsePhase
  const isThinking = responsePhase === 'pre_response';
  const hasResponse = responsePhase === 'in_response' || responsePhase === 'post_response';
  const isResponseStreaming = responsePhase === 'in_response';
  // no_tags or undefined: fall back to showing raw content

  // Header label suffix for thinking state
  const headerSuffix = isThinking && !isComplete ? ' · Thinking...' : '';

  return (
    <Paper
      elevation={0}
      sx={{
        mb: 1.5,
        backgroundColor: theme.custom.surfaces.highlightSubtle,
        borderRadius: 2,
        border: '1px solid',
        borderColor: isComplete ? 'success.main' : 'primary.dark',
        overflow: 'hidden',
        opacity: isComplete && collapsed ? 0.85 : 1,
        transition: 'opacity 0.2s',
      }}
    >
      {/* ProgressSection-style header bar */}
      <Box
        onClick={() => setCollapsed(!collapsed)}
        sx={{
          p: 1.5,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          cursor: 'pointer',
          backgroundColor: theme.custom.surfaces.sidebarBg,
          borderBottom: collapsed ? 'none' : `1px solid ${theme.custom.surfaces.overlayActive}`,
          '&:hover': { backgroundColor: theme.custom.surfaces.scrim },
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
            {displayLabel}
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
                backgroundColor: alpha(theme.palette.success.main, 0.15),
                color: 'success.light',
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
                  backgroundColor: theme.custom.surfaces.overlayActive,
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
                onViewAll(agentId, content);
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
                  backgroundColor: theme.custom.surfaces.overlayActive,
                },
              }}
            >
              View Response
            </Button>
          )}
          {showStatus && (isComplete ? (
            <CheckCircleIcon sx={{ color: 'success.main', fontSize: 18 }} />
          ) : (
            <CircularProgress size={14} thickness={4} />
          ))}
        </Box>
      </Box>

      {/* Content area with max height */}
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
          {/* Phase: pre_response — show thinking content in muted style */}
          {isThinking && (
            <>
              <Box sx={{ opacity: 0.5, color: 'text.secondary' }}>
                <MarkdownRenderer content={thinkingContent || ''} />
              </Box>
              {!isComplete && <BlinkingCursor />}
            </>
          )}

          {/* Phase: in_response or post_response — thinking fold + response */}
          {hasResponse && (
            <>
              <ThinkingFold thinkingContent={thinkingContent} />
              <MarkdownRenderer content={stripToolsToInvoke(responseContent || '')} />
              {isResponseStreaming && !isComplete && <BlinkingCursor />}
            </>
          )}

          {/* Phase: no_tags or undefined — fallback to raw content */}
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

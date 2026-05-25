/**
 * PromptViewerDrawer — right-side slide panel for inspecting prompt data
 * and viewing full AI response content.
 *
 * Two modes:
 * - Prompt mode (promptData provided): 3 tabs — Template, Variables, Rendered Prompt
 * - Full Response mode (fullResponseContent provided): full markdown response
 */

import React, { useState, useEffect, useMemo } from 'react';
import { parseResponseTags } from './ThinkingFold';
import {
  Box,
  Drawer,
  IconButton,
  Tab,
  Tabs,
  Typography,
} from '@mui/material';
import { Close as CloseIcon } from '@mui/icons-material';
import { MarkdownRenderer } from './MarkdownRenderer';

const PROMPT_TABS = ['Template', 'Variables', 'Rendered Prompt'];
const RESPONSE_TABS = ['Rendered', 'Raw'];

/** Monospaced preformatted text block */
function CodeBlock({ content }) {
  return (
    <Box
      sx={{
        fontFamily: 'monospace',
        fontSize: '0.78rem',
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
        color: 'text.secondary',
        backgroundColor: 'rgba(0,0,0,0.3)',
        p: 2,
        borderRadius: 1,
        overflow: 'auto',
      }}
    >
      {content || '(not available)'}
    </Box>
  );
}

/** Renders template feed key/value pairs */
function VariablesTab({ templateFeed, templateConfig }) {
  const feed = templateFeed || {};
  const config = templateConfig || {};

  return (
    <Box>
      {/* Template feed variables */}
      <Typography variant="overline" sx={{ color: 'text.disabled', display: 'block', mb: 1 }}>
        Template Variables
      </Typography>
      {Object.keys(feed).length === 0 && (
        <Typography variant="caption" sx={{ color: 'text.disabled' }}>
          (no variables available)
        </Typography>
      )}
      {Object.entries(feed).map(([key, value]) => {
        const str =
          typeof value === 'string'
            ? value
            : JSON.stringify(value, null, 2);
        // Truncate very large values
        const display = str.length > 3000 ? str.slice(0, 3000) + '\n\n... (truncated)' : str;
        return (
          <Box key={key} sx={{ mb: 2 }}>
            <Typography
              variant="caption"
              sx={{
                color: 'primary.light',
                fontWeight: 700,
                fontFamily: 'monospace',
                display: 'block',
                mb: 0.5,
              }}
            >
              {key}
            </Typography>
            <Box
              sx={{
                fontFamily: 'monospace',
                fontSize: '0.75rem',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                color: 'text.secondary',
                backgroundColor: 'rgba(0,0,0,0.2)',
                p: 1.5,
                borderRadius: 1,
                maxHeight: 220,
                overflow: 'auto',
              }}
            >
              {display}
            </Box>
          </Box>
        );
      })}

      {/* Template config (tools whitelist etc.) */}
      {Object.keys(config).length > 0 && (
        <>
          <Typography variant="overline" sx={{ color: 'text.disabled', display: 'block', mt: 2, mb: 1 }}>
            Template Config (.initial.config.yaml)
          </Typography>
          <CodeBlock content={JSON.stringify(config, null, 2)} />
        </>
      )}
    </Box>
  );
}

export function PromptViewerDrawer({ open, onClose, promptData, fullResponseContent }) {
  const [tab, setTab] = useState(0);
  const [responseTab, setResponseTab] = useState(0);
  const isFullResponse = Boolean(fullResponseContent) && !promptData;

  // Reset tabs every time the drawer opens — ensures users always start
  // on the first tab regardless of which mode (prompt or full response)
  // or which message they open. Triggered on open, not on mode change,
  // because isFullResponse stays true between consecutive Full Response opens.
  useEffect(() => { if (open) { setResponseTab(0); setTab(0); } }, [open]);

  return (
    <Drawer
      anchor="right"
      open={open}
      onClose={onClose}
      PaperProps={{
        sx: {
          width: { xs: '100%', sm: 660, md: 780 },
          display: 'flex',
          flexDirection: 'column',
          backgroundColor: 'background.default',
          borderLeft: '1px solid rgba(255,255,255,0.08)',
        },
      }}
    >
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          px: 2,
          py: 1.5,
          borderBottom: '1px solid rgba(255,255,255,0.08)',
          flexShrink: 0,
        }}
      >
        <Typography variant="subtitle1" sx={{ flex: 1, fontWeight: 600, fontSize: '0.95rem' }}>
          {isFullResponse ? '📄 Full Response' : '🔍 Prompt Inspector'}
        </Typography>
        <IconButton onClick={onClose} size="small" sx={{ color: 'text.secondary' }}>
          <CloseIcon fontSize="small" />
        </IconButton>
      </Box>

      {/* Tabs — prompt mode: Template/Variables/Rendered Prompt */}
      {!isFullResponse && (
        <Tabs
          value={tab}
          onChange={(_, v) => setTab(v)}
          sx={{
            borderBottom: '1px solid rgba(255,255,255,0.08)',
            px: 2,
            flexShrink: 0,
            minHeight: 40,
            '& .MuiTab-root': { minHeight: 40, fontSize: '0.8rem', py: 0 },
          }}
        >
          {PROMPT_TABS.map((label, i) => (
            <Tab key={label} label={label} value={i} />
          ))}
        </Tabs>
      )}

      {/* Tabs — full response mode: Rendered / Raw */}
      {isFullResponse && (
        <Tabs
          value={responseTab}
          onChange={(_, v) => setResponseTab(v)}
          sx={{
            borderBottom: '1px solid rgba(255,255,255,0.08)',
            px: 2,
            flexShrink: 0,
            minHeight: 40,
            '& .MuiTab-root': { minHeight: 40, fontSize: '0.8rem', py: 0 },
          }}
        >
          {RESPONSE_TABS.map((label, i) => (
            <Tab key={label} label={label} value={i} />
          ))}
        </Tabs>
      )}

      {/* Content */}
      <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
        {/* Full Response mode — two tabs: Rendered (markdown) + Raw (plain text) */}
        {isFullResponse && (
          <>
            {/* Tab 0: Rendered — strip <Response> tags for clean markdown view */}
            {responseTab === 0 && (() => {
              const parsed = parseResponseTags(fullResponseContent || '');
              const renderedContent = parsed.responseContent || fullResponseContent || '';
              return <MarkdownRenderer content={renderedContent} />;
            })()}
            {/* Tab 1: Raw — original text with all tags, useful for debugging */}
            {responseTab === 1 && (
              <CodeBlock content={fullResponseContent} />
            )}
          </>
        )}

        {/* Prompt mode */}
        {!isFullResponse && promptData && (
          <>
            {/* Tab 0: Template source (raw Jinja2) */}
            {tab === 0 && (
              <CodeBlock content={promptData.template_source} />
            )}

            {/* Tab 1: Variables (template feed + config) */}
            {tab === 1 && (
              <VariablesTab
                templateFeed={promptData.template_feed}
                templateConfig={promptData.template_config}
              />
            )}

            {/* Tab 2: Rendered prompt */}
            {tab === 2 && (
              <CodeBlock content={promptData.rendered_prompt} />
            )}
          </>
        )}

        {/* No data state */}
        {!isFullResponse && !promptData && (
          <Typography variant="body2" sx={{ color: 'text.disabled', mt: 2 }}>
            No prompt data available.
          </Typography>
        )}
      </Box>
    </Drawer>
  );
}

export default PromptViewerDrawer;

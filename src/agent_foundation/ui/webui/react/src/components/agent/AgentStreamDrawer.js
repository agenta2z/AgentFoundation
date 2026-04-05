/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * AgentStreamDrawer — Right-side drawer for viewing full agent output.
 *
 * When turnData is provided (from View Prompt), renders a tabbed view:
 *   Tab 1: Prompt Template & Feed (template source + variables table)
 *   Tab 2: Rendered Prompt (fully expanded Jinja2 output)
 *   Tab 3: API Payload (actual system_prompt + messages sent to the LLM)
 *
 * When turnData is absent, falls back to the original single-content view
 * (used by "View All" and other non-prompt drawer uses).
 */

import React, { useState } from 'react';
import {
  Box,
  Drawer,
  Typography,
  IconButton,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { Close as CloseIcon } from '@mui/icons-material';
import { MarkdownRenderer } from '../common/MarkdownRenderer';
import { stripResponseTags, stripToolsToInvoke } from './AgentStreamSection';

const AGENT_LABELS = {
  base: 'Base Agent',
  review: 'Review Agent',
  system: 'System',
};

const getContentBoxSx = (theme) => ({
  flex: 1,
  overflow: 'auto',
  backgroundColor: theme.custom.surfaces.sidebarBg,
  borderRadius: 1,
  p: 2,
  height: 'calc(100vh - 160px)',
  '& p': { m: 0 },
  '& pre': { overflow: 'auto' },
});

function TabPanel({ children, value, index }) {
  if (value !== index) return null;
  return <Box sx={{ height: '100%' }}>{children}</Box>;
}

const ROLE_PALETTE = { user: 'success', assistant: 'info', system: 'warning' };

/** Tab 1: Prompt Template source + Feed variables table */
function TemplateAndFeedTab({ turnData }) {
  const theme = useTheme();
  const contentBoxSx = getContentBoxSx(theme);
  const template = turnData?.prompt_template || '';
  const feed = turnData?.template_feed;

  // Build feed rows — flatten the feed dict into label/value pairs.
  // For large values (conversation_history, available_tools), show a truncated preview.
  const feedRows = [];
  if (feed && typeof feed === 'object') {
    for (const [key, value] of Object.entries(feed)) {
      let display;
      if (typeof value === 'string') {
        display = value.length > 300 ? value.slice(0, 300) + '...' : value;
      } else {
        const json = JSON.stringify(value, null, 2);
        display = json.length > 300 ? json.slice(0, 300) + '...' : json;
      }
      feedRows.push({ key, value: display, fullValue: typeof value === 'string' ? value : JSON.stringify(value, null, 2) });
    }
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', gap: 2 }}>
      {/* Template Source */}
      <Box sx={{ flex: '0 0 auto', maxHeight: '45%', overflow: 'auto' }}>
        <Typography variant="subtitle2" sx={{ mb: 1, color: 'text.secondary' }}>
          Prompt Template
        </Typography>
        <Box sx={{ ...contentBoxSx, height: 'auto', maxHeight: '100%' }}>
          {template ? (
            <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: '0.8rem' }}>
              {template}
            </pre>
          ) : (
            <Typography variant="body2" color="text.secondary">
              No template source available (fallback prompt was used).
            </Typography>
          )}
        </Box>
      </Box>

      {/* Feed Variables Table */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        <Typography variant="subtitle2" sx={{ mb: 1, color: 'text.secondary' }}>
          Template Feed Variables
        </Typography>
        {feedRows.length > 0 ? (
          <TableContainer component={Paper} sx={{ backgroundColor: theme.custom.surfaces.sidebarBg, maxHeight: '100%' }}>
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 'bold', backgroundColor: theme.custom.surfaces.scrim, color: 'text.primary', width: '20%' }}>
                    Variable
                  </TableCell>
                  <TableCell sx={{ fontWeight: 'bold', backgroundColor: theme.custom.surfaces.scrim, color: 'text.primary' }}>
                    Value
                  </TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {feedRows.map((row) => (
                  <TableRow key={row.key} hover>
                    <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.8rem', verticalAlign: 'top', color: 'primary.light' }}>
                      {row.key}
                    </TableCell>
                    <TableCell>
                      <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: '0.75rem', maxHeight: 200, overflow: 'auto' }}>
                        {row.value}
                      </pre>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        ) : (
          <Typography variant="body2" color="text.secondary">
            No feed variables available.
          </Typography>
        )}
      </Box>
    </Box>
  );
}

/** Tab 3: API Payload — system_prompt + messages as sent to the LLM */
function ApiPayloadTab({ turnData }) {
  const theme = useTheme();
  const contentBoxSx = getContentBoxSx(theme);
  const payload = turnData?.api_payload;

  if (!payload) {
    return (
      <Box sx={contentBoxSx}>
        <Typography variant="body2" color="text.secondary">
          No API payload data available for this turn.
        </Typography>
      </Box>
    );
  }

  const systemPrompt = typeof payload === 'object' ? payload.system_prompt : '';
  const messages = typeof payload === 'object' ? (payload.messages || []) : [];

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', gap: 2 }}>
      {/* System Prompt */}
      <Box sx={{ flex: '0 0 auto', maxHeight: '40%', overflow: 'auto' }}>
        <Typography variant="subtitle2" sx={{ mb: 1, color: 'text.secondary' }}>
          System Prompt
        </Typography>
        <Box sx={{ ...contentBoxSx, height: 'auto', maxHeight: '100%' }}>
          <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: '0.8rem' }}>
            {systemPrompt || '(empty)'}
          </pre>
        </Box>
      </Box>

      {/* Messages */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        <Typography variant="subtitle2" sx={{ mb: 1, color: 'text.secondary' }}>
          Messages ({messages.length})
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          {messages.map((msg, i) => (
            <Box
              key={i}
              sx={{
                ...contentBoxSx,
                height: 'auto',
                maxHeight: 300,
                borderLeft: `3px solid ${theme.palette[ROLE_PALETTE[msg.role] || 'primary'].main}`,
              }}
            >
              <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 'bold', textTransform: 'uppercase' }}>
                {msg.role}
              </Typography>
              <pre style={{ margin: '4px 0 0', whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: '0.8rem' }}>
                {msg.content}
              </pre>
            </Box>
          ))}
        </Box>
      </Box>
    </Box>
  );
}

export function AgentStreamDrawer({ open, onClose, agentId, content, turnData }) {
  const theme = useTheme();
  const contentBoxSx = getContentBoxSx(theme);
  const [tabIndex, setTabIndex] = useState(0);

  // When turnData is present, show tabbed prompt viewer
  const hasTurnData = turnData && (turnData.rendered_prompt || turnData.prompt_template || turnData.api_payload);

  // Reset tab when drawer opens with new data
  React.useEffect(() => {
    if (open) {
      // Default to Tab 2 (Rendered Prompt) if no template data, else Tab 1
      if (turnData?.prompt_template || turnData?.template_feed) {
        setTabIndex(0);
      } else if (turnData?.rendered_prompt) {
        setTabIndex(1);
      } else {
        setTabIndex(0);
      }
    }
  }, [open, turnData]);

  return (
    <Drawer
      anchor="right"
      open={open}
      onClose={onClose}
      PaperProps={{
        sx: { width: { xs: '100%', sm: 600, md: 900 }, p: 2 },
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
        <Typography variant="h6" sx={{ flex: 1 }}>
          {AGENT_LABELS[agentId] || agentId}{hasTurnData ? '' : ' — Full Output'}
        </Typography>
        <IconButton onClick={onClose}>
          <CloseIcon />
        </IconButton>
      </Box>

      {hasTurnData ? (
        /* Tabbed prompt viewer */
        <Box sx={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 100px)' }}>
          <Tabs
            value={tabIndex}
            onChange={(_, v) => setTabIndex(v)}
            sx={{
              minHeight: 36,
              '& .MuiTab-root': { minHeight: 36, py: 0.5, textTransform: 'none', fontSize: '0.85rem' },
            }}
          >
            <Tab label="Template & Feed" />
            <Tab label="Rendered Prompt" />
            <Tab label="API Payload" />
          </Tabs>
          <Box sx={{ flex: 1, overflow: 'auto', mt: 1 }}>
            <TabPanel value={tabIndex} index={0}>
              <TemplateAndFeedTab turnData={turnData} />
            </TabPanel>
            <TabPanel value={tabIndex} index={1}>
              <Box sx={contentBoxSx}>
                <MarkdownRenderer
                  content={turnData?.rendered_prompt || 'No rendered prompt available.'}
                  structuralXmlTags={turnData?.template_config?.rendering?.structural_xml_tags}
                />
              </Box>
            </TabPanel>
            <TabPanel value={tabIndex} index={2}>
              <ApiPayloadTab turnData={turnData} />
            </TabPanel>
          </Box>
        </Box>
      ) : (
        /* Original single-content view (for View All, etc.) */
        <Box sx={contentBoxSx}>
          <MarkdownRenderer content={stripToolsToInvoke(stripResponseTags(content || ''))} />
        </Box>
      )}
    </Drawer>
  );
}

export default AgentStreamDrawer;

/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * RunStreamPanel — Generic right-panel streaming display for the selected run.
 * Shows live streaming sections or snapshotted completed sections.
 *
 * Ported verbatim from RankEvolve (components/views/RunStreamPanel.js); only
 * the AgentStreamSection import path changed (local to this dashboard).
 */

import React, { useRef, useEffect, useCallback } from 'react';
import { Box, Typography } from '@mui/material';
import { AgentStreamSection } from './AgentStreamSection';

export default function RunStreamPanel({
  sections,
  isStreaming,
  runLabel,
  runDescription,
  runStatus,
}) {
  const scrollRef = useRef(null);
  const bottomRef = useRef(null);

  const isNearBottom = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return true;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 150;
  }, []);

  useEffect(() => {
    if (isNearBottom()) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [sections, isNearBottom]);

  return (
    <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {runLabel && (
        <Box sx={{
          px: 2, py: 1,
          borderBottom: '1px solid', borderColor: 'divider',
          backgroundColor: 'rgba(0, 0, 0, 0.05)',
        }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 600, fontSize: '0.85rem' }}>
            {runLabel}
          </Typography>
          {runDescription && (
            <Typography variant="caption" color="text.secondary">
              {runDescription}
            </Typography>
          )}
        </Box>
      )}

      <Box ref={scrollRef} sx={{ flex: 1, overflow: 'auto', px: 1, py: 2 }}>
        {(!sections || sections.length === 0) && !isStreaming && (
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', opacity: 0.5 }}>
            <Typography variant="body2" color="text.secondary">
              {runStatus === 'completed' ? 'Run completed.' :
               runStatus === 'error' ? 'Run failed.' :
               'Waiting for output...'}
            </Typography>
          </Box>
        )}

        {(sections || []).map((section, idx) => (
          <AgentStreamSection
            key={`${section.agentId}-${idx}`}
            agentId={section.agentId}
            content={section.content}
            isComplete={section.isComplete}
            turnNumber={section.turnNumber || (idx + 1)}
            thinkingContent={section.thinkingContent}
            responseContent={section.responseContent}
            responsePhase={section.responsePhase}
          />
        ))}

        {isStreaming && (!sections || sections.length === 0) && (
          <AgentStreamSection
            agentId="system"
            content="*Thinking (may take a few minutes)...*"
            isComplete={false}
            isPlaceholder
          />
        )}

        <div ref={bottomRef} />
      </Box>
    </Box>
  );
}

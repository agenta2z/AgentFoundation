/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * QueueProgressView — Split-panel multi-run queue host.
 * Left panel: RunQueueList (clickable queue navigator)
 * Right panel: RunStreamPanel (streaming output for selected run)
 *
 * Ported from RankEvolve (components/views/QueueProgressView.js). In the
 * decoupled dashboard the live streaming sections for a sub-task are read
 * from the reducer state (`task.liveSections[subTaskId]`, accumulated from
 * `wsEvents$`) instead of from a SessionContext streaming ref; completed
 * sections come from `entry.completedSections` (reducer-snapshotted). The
 * `resumeImplementTask` apiClient action is threaded into RunQueueList.
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Accordion, AccordionDetails, AccordionSummary,
  Box, Chip, LinearProgress, Typography,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import RunQueueList from './RunQueueList';
import RunStreamPanel from './RunStreamPanel';
import { MarkdownRenderer } from '../../../../common/MarkdownRenderer';

/**
 * Lazily fetches a markdown file from the sub-task's workspace via
 * GET /api/workspace/file?workspace=<abs>&path=<rel>. Renders inline.
 */
function LazyWorkspaceMarkdown({ workspace, relativePath, emptyHint }) {
  const [content, setContent] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    if (!workspace || !relativePath) return undefined;
    let cancelled = false;
    fetch(
      '/api/workspace/file'
      + `?workspace=${encodeURIComponent(workspace)}`
      + `&path=${encodeURIComponent(relativePath)}`,
    )
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((body) => {
        if (cancelled) return;
        const c = typeof body?.content === 'string'
          ? body.content
          : (body?.content ? JSON.stringify(body.content, null, 2) : '');
        setContent(c);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e.message);
      });
    return () => { cancelled = true; };
  }, [workspace, relativePath]);
  if (error) {
    return (
      <Typography variant="caption" sx={{ color: 'text.secondary', fontStyle: 'italic' }}>
        {emptyHint || `Couldn't load ${relativePath}: ${error}`}
      </Typography>
    );
  }
  if (content === null) {
    return (
      <Typography variant="caption" sx={{ color: 'text.secondary' }}>
        Loading {relativePath}…
      </Typography>
    );
  }
  if (!content) {
    return (
      <Typography variant="caption" sx={{ color: 'text.secondary', fontStyle: 'italic' }}>
        {emptyHint || `${relativePath} is empty.`}
      </Typography>
    );
  }
  return <MarkdownRenderer content={content} />;
}

function BatchLabel({ entry }) {
  if (!entry.metadata?.batchId) {
    return (
      <Typography variant="body2" sx={{ fontSize: '0.8rem', fontWeight: 600 }}>
        {entry.label || 'Task'}
      </Typography>
    );
  }
  return (
    <Box>
      <Typography variant="body2" sx={{ fontSize: '0.8rem', fontWeight: 600 }}>
        B{entry.metadata.batchId}: {entry.metadata.batchLabel || ''}
      </Typography>
      <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.7rem' }}>
        {(entry.metadata.hypothesisIds || []).join(', ')}
      </Typography>
    </Box>
  );
}

export default function QueueProgressView({
  task,
  config,
  dispatch,
  apiClient,
}) {
  const queue = task.runQueue || [];
  const currentRunIndex = task.currentRunIndex ?? -1;
  const [selectedIndex, setSelectedIndex] = useState(currentRunIndex >= 0 ? currentRunIndex : 0);
  const [autoFollow, setAutoFollow] = useState(config.autoFollow !== false);

  // Auto-follow: when a new run starts, advance selection to it
  useEffect(() => {
    if (autoFollow && currentRunIndex >= 0) {
      setSelectedIndex(currentRunIndex);
    }
  }, [currentRunIndex, autoFollow]);

  const handleSelect = useCallback((index) => {
    setSelectedIndex(index);
    setAutoFollow(index === currentRunIndex);
    dispatch({ type: 'SELECT_QUEUE_ENTRY', taskId: task.id, runIndex: index });
  }, [currentRunIndex, dispatch, task.id]);

  // Compute progress
  const completedCount = queue.filter(e => e.status === 'completed' || e.status === 'error').length;
  const progress = queue.length > 0 ? (completedCount / queue.length) * 100 : 0;
  const runningCount = queue.filter(e => e.status === 'running').length;
  const queuedCount = queue.filter(e => e.status === 'queued').length;

  // Get sections for the selected run. Live sections for an in-flight sub-task
  // are accumulated into the reducer's `task.liveSections[subTaskId]` from
  // the WS event bus; completed runs surface their snapshotted
  // `entry.completedSections`.
  const selectedEntry = queue[selectedIndex];
  const isSelectedRunning = selectedEntry?.subTaskId
    && (selectedEntry.status === 'running' || selectedEntry.status === 'starting');
  const liveSections = (selectedEntry?.subTaskId
    && task.liveSections && task.liveSections[selectedEntry.subTaskId]) || null;

  const displaySections = isSelectedRunning && liveSections && liveSections.length > 0
    ? liveSections
    : (selectedEntry?.completedSections || []);
  const displayIsStreaming = !!(isSelectedRunning
    && (!liveSections || liveSections.length === 0
      || liveSections.some(s => !s.isComplete)));

  // Render label based on config
  const renderLabel = config.renderLabel === 'batch'
    ? (entry) => <BatchLabel entry={entry} />
    : undefined;

  // Single entry flat mode: render without left panel
  const singleFlat = config.singleEntryFlat && queue.length <= 1;

  const currentLabel = selectedEntry?.label || '';
  const currentDescription = selectedEntry?.description || '';

  return (
    <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Overall Progress Bar */}
      <Box sx={{
        px: 2, py: 1,
        borderBottom: '1px solid', borderColor: 'divider',
        backgroundColor: 'rgba(0, 0, 0, 0.05)',
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
          <Typography variant="body2" sx={{ fontSize: '0.82rem', fontWeight: 600 }}>
            {currentRunIndex >= 0
              ? `Run ${currentRunIndex + 1} of ${queue.length}: ${currentLabel}`
              : `${queue.length} runs queued`}
          </Typography>
        </Box>
        <LinearProgress
          variant="determinate"
          value={progress}
          sx={{ height: 6, borderRadius: 3, mb: 0.5 }}
        />
        <Box sx={{ display: 'flex', gap: 1 }}>
          {completedCount > 0 && (
            <Chip label={`✅ ${completedCount} done`} size="small" color="success" variant="outlined"
              sx={{ height: 18, fontSize: '0.6rem' }} />
          )}
          {runningCount > 0 && (
            <Chip label={`▶ ${runningCount} running`} size="small" color="info" variant="outlined"
              sx={{ height: 18, fontSize: '0.6rem' }} />
          )}
          {queuedCount > 0 && (
            <Chip label={`⏳ ${queuedCount} queued`} size="small" variant="outlined"
              sx={{ height: 18, fontSize: '0.6rem' }} />
          )}
        </Box>
      </Box>

      {/* Split Panel */}
      <Box sx={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {!singleFlat && (
          <RunQueueList
            runQueue={queue}
            selectedIndex={selectedIndex}
            currentRunIndex={currentRunIndex}
            onSelect={handleSelect}
            renderLabel={renderLabel}
            resumeImplementTask={apiClient && apiClient.resumeImplementTask}
          />
        )}

        <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <Box sx={{ flex: displaySections?.length || displayIsStreaming || selectedEntry?.status !== 'completed' ? 1 : 0, overflow: 'hidden' }}>
            <RunStreamPanel
              sections={displaySections}
              isStreaming={displayIsStreaming}
              runLabel={singleFlat ? undefined : currentLabel}
              runDescription={singleFlat ? undefined : currentDescription}
              runStatus={selectedEntry?.status}
            />
          </Box>

          {selectedEntry?.status === 'completed'
            && (!displaySections || displaySections.length === 0)
            && selectedEntry?.workspacePath && (
            <Box sx={{ flex: 1, overflow: 'auto', p: 2, borderTop: '1px solid rgba(255,255,255,0.08)' }}>
              <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 1 }}>
                Run completed — workspace artifacts:
              </Typography>
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>summary.md</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <LazyWorkspaceMarkdown
                    workspace={selectedEntry.workspacePath}
                    relativePath="summary.md"
                    emptyHint="(no summary.md written for this batch)"
                  />
                </AccordionDetails>
              </Accordion>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>outputs/round0_plan.md</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <LazyWorkspaceMarkdown
                    workspace={selectedEntry.workspacePath}
                    relativePath="outputs/round0_plan.md"
                    emptyHint="(no plan output written)"
                  />
                </AccordionDetails>
              </Accordion>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>outputs/round0_implementation.md</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <LazyWorkspaceMarkdown
                    workspace={selectedEntry.workspacePath}
                    relativePath="outputs/round0_implementation.md"
                    emptyHint="(no implementation output written)"
                  />
                </AccordionDetails>
              </Accordion>
            </Box>
          )}
        </Box>
      </Box>
    </Box>
  );
}

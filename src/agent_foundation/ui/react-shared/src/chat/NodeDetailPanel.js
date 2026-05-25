/**
 * NodeDetailPanel — shows streaming content for the selected graph node.
 *
 * Shows live streaming content while the node is running, and the full
 * final output for completed nodes. Auto-scrolls as content arrives.
 *
 * Props:
 *   node        - node object {id, label, status, startedAt, completedAt}
 *   content     - accumulated stream content (from nodeStreams[selectedNodeId])
 *   isStreaming  - true when node.status === 'running' and tokens are arriving
 */

import React, { useEffect, useRef, useState, useMemo } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Chip from '@mui/material/Chip';
import { stripAnsi, stripAcliNoise } from './ThinkingFold';
import CircularProgress from '@mui/material/CircularProgress';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import { useTheme } from '@mui/material/styles';
import { MarkdownRenderer } from './MarkdownRenderer';

/**
 * Fetch file content from /api/view/{absolutePath}.
 * Returns { content, loading, error }.
 * The /api/view endpoint serves _runtime files only (security-checked).
 */
function useNodeOutput(outputPath) {
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!outputPath) { setContent(''); setLoading(false); setError(null); return; }
    let cancelled = false;
    setLoading(true);
    setContent('');
    setError(null);
    fetch(`/api/view/${outputPath}`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.text();
      })
      .then(text => { if (!cancelled) { setContent(text); setLoading(false); } })
      .catch(err => { if (!cancelled) { setError(err.message); setLoading(false); } });
    return () => { cancelled = true; };
  }, [outputPath]);

  return { content, loading, error };
}

const STATUS_CONFIG = {
  pending:   { label: 'Pending',  color: 'default',  showSpinner: false },
  running:   { label: 'Running',  color: 'info',     showSpinner: true  },
  completed: { label: 'Complete', color: 'success',  showSpinner: false },
  error:     { label: 'Error',    color: 'error',    showSpinner: false },
  skipped:   { label: 'Skipped',  color: 'warning',  showSpinner: false },
};

function ElapsedTimer({ startedAt, completedAt }) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!startedAt) return;
    if (completedAt) {
      setElapsed(Math.round(completedAt - startedAt));
      return;
    }
    const iv = setInterval(() => {
      setElapsed(Math.round(Date.now() / 1000 - startedAt));
    }, 1000);
    return () => clearInterval(iv);
  }, [startedAt, completedAt]);

  if (!startedAt) return null;
  return (
    <Typography variant="caption" sx={{ color: 'text.disabled', ml: 1 }}>
      {elapsed}s
    </Typography>
  );
}

export function NodeDetailPanel({ node, content: streamContent, isStreaming }) {
  const theme = useTheme();
  const bottomRef = useRef(null);

  // For completed nodes: fetch output file via /api/view/{outputPath}
  // For running/pending: show streamed content (or "running..." placeholder)
  const isCompleted = node?.status === 'completed' || node?.status === 'skipped';
  const { content: fileContent, loading: fileLoading, error: fileError } =
    useNodeOutput(isCompleted ? (node?.outputPath || '') : '');

  // Displayed content: completed nodes prefer fetched file, fall back to stream
  const rawContent = isCompleted ? (fileContent || streamContent) : streamContent;
  // Cosmetic: strip terminal escape codes and ACLI noise for cleaner display
  // (keeps all actual content — tool calls, outputs, etc.)
  const content = useMemo(() => stripAcliNoise(stripAnsi(rawContent || '')), [rawContent]);
  const loading = isCompleted && fileLoading;

  // Auto-scroll as content arrives
  useEffect(() => {
    if (isStreaming) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [streamContent, isStreaming]);

  if (!node) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', p: 3 }}>
        <Typography variant="body2" sx={{ color: 'text.secondary' }}>
          Click a node in the graph above to see its output.
        </Typography>
      </Box>
    );
  }

  const cfg = STATUS_CONFIG[node.status] || STATUS_CONFIG.pending;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Header */}
      <Box
        sx={{
          display: 'flex', alignItems: 'center', gap: 1,
          px: 2, py: 1,
          borderBottom: '1px solid rgba(255,255,255,0.06)',
          flexShrink: 0,
        }}
      >
        {cfg.showSpinner && <CircularProgress size={14} color="info" />}
        {!cfg.showSpinner && node.status === 'completed' && (
          <CheckCircleIcon sx={{ fontSize: 16, color: 'success.main' }} />
        )}
        {!cfg.showSpinner && node.status === 'error' && (
          <ErrorIcon sx={{ fontSize: 16, color: 'error.main' }} />
        )}

        <Typography variant="subtitle2" sx={{ fontWeight: 600, flex: 1,
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
          title={node.label}
        >
          {node.label}
        </Typography>

        <Chip
          label={cfg.label}
          color={cfg.color}
          size="small"
          variant="outlined"
          sx={{ height: 20, fontSize: '0.7rem', '& .MuiChip-label': { px: 0.75 } }}
        />
        <ElapsedTimer startedAt={node.startedAt} completedAt={node.completedAt} />
      </Box>

      {/* Content area */}
      <Box sx={{ flex: 1, overflow: 'auto', px: 2, py: 1.5 }}>
        {loading && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, py: 2 }}>
            <CircularProgress size={14} />
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>Loading output…</Typography>
          </Box>
        )}

        {fileError && !loading && (
          <Typography variant="body2" sx={{ color: 'error.main', fontStyle: 'italic' }}>
            Could not load output: {fileError}
          </Typography>
        )}

        {content && !loading ? (
          <Box sx={{ '& p': { mt: 0, mb: 1 }, '& pre': { overflow: 'auto' } }}>
            <MarkdownRenderer content={content} />
          </Box>
        ) : !loading && !fileError && node.status === 'pending' ? (
          <Typography variant="body2" sx={{ color: 'text.disabled', fontStyle: 'italic' }}>
            Waiting to start…
          </Typography>
        ) : !loading && !fileError && node.status === 'running' ? (
          <Typography
            variant="body2"
            sx={{
              color: 'text.secondary', fontStyle: 'italic',
              animation: 'pulse 1.5s ease-in-out infinite',
              '@keyframes pulse': { '0%,100%': { opacity: 0.4 }, '50%': { opacity: 1 } },
            }}
          >
            {node.label} is running…
          </Typography>
        ) : !loading && !fileError && node.status === 'error' ? (
          <Typography variant="body2" sx={{ color: 'error.main' }}>
            {node.error || 'An error occurred.'}
          </Typography>
        ) : !loading && !content && !fileError ? (
          <Typography variant="body2" sx={{ color: 'text.disabled', fontStyle: 'italic' }}>
            {isCompleted ? 'Output not found.' : 'No output captured.'}
          </Typography>
        ) : null}

        {/* Blinking cursor while streaming */}
        {isStreaming && (
          <Box
            component="span"
            sx={{
              display: 'inline-block', width: 7, height: 15,
              backgroundColor: 'primary.main', ml: 0.5, verticalAlign: 'middle',
              animation: 'blink 1s step-end infinite',
              '@keyframes blink': { '50%': { opacity: 0 } },
            }}
          />
        )}
        <div ref={bottomRef} />
      </Box>
    </Box>
  );
}

export default NodeDetailPanel;

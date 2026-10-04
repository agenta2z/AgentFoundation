/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * LazyAnalysisDoc — Always shows an analysisSummary (1-paragraph synopsis).
 * On expand, lazy-fetches the full outputs/analysis.md from the GET
 * /api/view/{file_path:path} endpoint and renders it inside a scrollable
 * container. Cached after first fetch.
 *
 * Ported from RankEvolve (components/views/LazyAnalysisDoc.js); only the
 * MarkdownRenderer import path changed (shared AgentFoundation common dir).
 */

import React, { useState } from 'react';
import {
  Box, Button, Collapse, CircularProgress, Typography,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import { MarkdownRenderer } from '../../../../common/MarkdownRenderer';

export default function LazyAnalysisDoc({ summary, analysisFile, fallbackInline }) {
  const [expanded, setExpanded] = useState(false);
  const [fullContent, setFullContent] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleToggle = async () => {
    if (expanded) {
      setExpanded(false);
      return;
    }
    setExpanded(true);
    if (fullContent !== null || !analysisFile) return;
    setLoading(true);
    setError(null);
    try {
      const safe = analysisFile.split('/').map(encodeURIComponent).join('/');
      const res = await fetch(`/api/view/${safe}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      setFullContent(text);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  const inline = summary || fallbackInline || '';

  return (
    <Box sx={{ mt: 1, fontSize: '0.82rem', opacity: 0.92 }}>
      {inline && <MarkdownRenderer content={inline} />}
      {analysisFile && (
        <Button
          size="small"
          startIcon={expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
          onClick={handleToggle}
          sx={{ textTransform: 'none', fontSize: '0.72rem', mt: 0.5, p: 0.5 }}
        >
          {expanded ? 'Collapse full analysis' : 'Expand full analysis'}
        </Button>
      )}
      <Collapse in={expanded} timeout="auto">
        <Box sx={{
          mt: 1, maxHeight: '60vh', overflow: 'auto',
          border: '1px solid rgba(255,255,255,0.1)', borderRadius: 1, p: 1.5,
          backgroundColor: 'rgba(255,255,255,0.02)',
        }}>
          {loading && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <CircularProgress size={16} />
              <Typography variant="body2">Loading full analysis…</Typography>
            </Box>
          )}
          {error && (
            <Typography color="error" variant="body2">
              Failed to load full analysis: {error}
            </Typography>
          )}
          {!loading && !error && fullContent !== null && (
            <MarkdownRenderer content={fullContent} />
          )}
        </Box>
      </Collapse>
    </Box>
  );
}

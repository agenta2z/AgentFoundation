/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * HypothesisDetailDrawer — Slide-out drawer for reviewing a hypothesis's
 * implementation details: plan summary, implementation summary, consensus
 * history.
 *
 * Ported verbatim from RankEvolve (components/views/HypothesisDetailDrawer.js).
 */

import React from 'react';
import { Box, Drawer, Typography, Divider, Button, Chip, Accordion, AccordionSummary, AccordionDetails } from '@mui/material';
import { Close as CloseIcon, ExpandMore as ExpandMoreIcon } from '@mui/icons-material';

export default function HypothesisDetailDrawer({
  open,
  onClose,
  hypothesisId,
  task,
  proposalDetails,
}) {
  if (!hypothesisId) return null;

  const detail = proposalDetails?.[hypothesisId] || {};
  const result = task?.scenarioState?.hypothesisResults?.[hypothesisId] || {};

  return (
    <Drawer anchor="right" open={open} onClose={onClose} PaperProps={{ sx: { width: 500, maxWidth: '90vw' } }}>
      <Box sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
          <Box>
            <Typography variant="h6" sx={{ fontWeight: 700, fontSize: '1rem' }}>
              <Typography component="span" sx={{ color: 'primary.main', fontWeight: 700, mr: 0.5 }}>
                {hypothesisId}
              </Typography>
              {detail.title || hypothesisId}
            </Typography>
            {detail.theme && (
              <Typography variant="body2" sx={{ color: 'text.secondary', mt: 0.3 }}>
                {detail.theme}
              </Typography>
            )}
          </Box>
          <Button size="small" onClick={onClose} sx={{ minWidth: 0 }}>
            <CloseIcon />
          </Button>
        </Box>

        <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
          {detail.impact && <Chip label={detail.impact} size="small" variant="outlined" />}
          {detail.complexity && <Chip label={detail.complexity} size="small" variant="outlined" />}
          {result.status && (
            <Chip
              label={result.status === 'completed' ? '✅ Implemented' : '❌ Failed'}
              size="small"
              color={result.status === 'completed' ? 'success' : 'error'}
              variant="outlined"
            />
          )}
          {(detail._isComboHypothesis
            || (Array.isArray(detail.includes) && detail.includes.length > 0)) && (
            <Chip
              label={`🔗 Combo alias of ${(detail._componentIds || detail.includes || []).join(', ')}`}
              size="small"
              color="warning"
              variant="outlined"
              title={
                detail._comboReasonAnnotated
                  ? `${detail._deprioritizeReason} — comboKey="${detail._comboKey}"`
                  : `Pre-bundles: ${(detail._componentIds || detail.includes || []).join(', ')}. Use the matching combo on the Review & Combos tab.`
              }
            />
          )}
        </Box>

        <Divider sx={{ mb: 2 }} />

        {detail.problem && (
          <Box sx={{ mb: 2 }}>
            <Typography sx={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', color: 'text.secondary', mb: 0.5 }}>
              Problem
            </Typography>
            <Typography variant="body2" sx={{ fontSize: '0.85rem' }}>
              {detail.problem}
            </Typography>
          </Box>
        )}
        {detail.approach && (
          <Box sx={{ mb: 2 }}>
            <Typography sx={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', color: 'text.secondary', mb: 0.5 }}>
              Approach
            </Typography>
            <Typography variant="body2" sx={{ fontSize: '0.85rem' }}>
              {detail.approach}
            </Typography>
          </Box>
        )}

        <Divider sx={{ mb: 2 }} />

        {result.planSummary && (
          <Box sx={{ mb: 2 }}>
            <Typography sx={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', color: 'text.secondary', mb: 0.5 }}>
              Plan Summary
            </Typography>
            <Typography variant="body2" sx={{
              fontSize: '0.82rem',
              backgroundColor: 'rgba(255,255,255,0.03)',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: 1, p: 1.5,
              whiteSpace: 'pre-wrap',
            }}>
              {result.planSummary}
            </Typography>
          </Box>
        )}

        {result.implSummary && (
          <Box sx={{ mb: 2 }}>
            <Typography sx={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', color: 'text.secondary', mb: 0.5 }}>
              Implementation Summary
            </Typography>
            <Typography variant="body2" sx={{
              fontSize: '0.82rem',
              backgroundColor: 'rgba(255,255,255,0.03)',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: 1, p: 1.5,
              whiteSpace: 'pre-wrap',
            }}>
              {result.implSummary}
            </Typography>
          </Box>
        )}

        {result.experimentList && result.experimentList.length > 0 && (
          <Accordion defaultExpanded sx={{ backgroundColor: 'transparent', boxShadow: 'none', mb: 1 }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography sx={{ fontSize: '0.82rem', fontWeight: 600 }}>
                Experiments this hypothesis appears in ({result.experimentList.length})
              </Typography>
            </AccordionSummary>
            <AccordionDetails sx={{ p: 0 }}>
              {result.experimentList.map((exp, i) => (
                <Box key={exp.id || i} sx={{
                  mb: 0.5, p: 1,
                  border: '1px solid rgba(255,255,255,0.08)',
                  borderRadius: 1,
                  backgroundColor: 'rgba(255,255,255,0.02)',
                }}>
                  <Typography sx={{ fontSize: '0.78rem', fontWeight: 600 }}>
                    <Typography component="span" sx={{
                      color: 'primary.main', fontWeight: 700, mr: 0.5,
                    }}>
                      {exp.comboKey || exp.id}
                    </Typography>
                    {exp.label}
                  </Typography>
                  {exp.workspace && (
                    <Typography sx={{ fontSize: '0.7rem', color: 'text.secondary', mt: 0.2 }}>
                      Workspace: {exp.workspace}
                    </Typography>
                  )}
                </Box>
              ))}
            </AccordionDetails>
          </Accordion>
        )}

        {result.reviewSummaryPath && (
          <Box sx={{ mb: 2 }}>
            <Typography sx={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', color: 'text.secondary', mb: 0.5 }}>
              Current Evaluation Summary (auto-generated placeholder)
            </Typography>
            <Typography variant="body2" sx={{
              fontSize: '0.78rem',
              fontFamily: 'monospace',
              color: 'text.secondary',
              backgroundColor: 'rgba(255,255,255,0.02)',
              border: '1px dashed rgba(255,255,255,0.1)',
              borderRadius: 1, p: 1,
            }}>
              {result.reviewSummaryPath}
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5, fontSize: '0.7rem', fontStyle: 'italic' }}>
              A future hypothesis-review task will replace this placeholder with a deeper qualitative + quantitative analysis.
            </Typography>
          </Box>
        )}

        {result.consensusInfo && (
          <Accordion sx={{ backgroundColor: 'transparent', boxShadow: 'none' }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography sx={{ fontSize: '0.82rem', fontWeight: 600 }}>
                Consensus History ({result.consensusInfo.total_iterations || '?'} iterations)
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2" sx={{ fontSize: '0.78rem', whiteSpace: 'pre-wrap' }}>
                {result.consensusInfo.consensus_achieved ? '✅ Consensus achieved' : '⚠️ No consensus'}
                {'\n'}Iterations: {result.consensusInfo.total_iterations || 'N/A'}
              </Typography>
            </AccordionDetails>
          </Accordion>
        )}

        {result.workspacePath && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="caption" color="text.secondary">
              Workspace: {result.workspacePath}
            </Typography>
          </Box>
        )}
      </Box>
    </Drawer>
  );
}

/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * ComboReviewSection — Active & Pending combos card-stack rendered in the
 * Review & Combo tab (MultiChoiceComboView). Each card is foldable,
 * multi-selectable, and removable; visual treatment reflects applyState.
 *
 * Ported verbatim from RankEvolve (components/views/ComboReviewSection.js).
 */

import React from 'react';
import {
  Box, Card, CardContent, Typography, Chip, Tooltip, Button,
  Checkbox, Collapse, IconButton, Stack,
} from '@mui/material';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import BuildIcon from '@mui/icons-material/Build';
import VpnKeyOffIcon from '@mui/icons-material/VpnKeyOff';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';

function ComboCard({
  combo,
  checked,
  expanded,
  onToggleCheck,
  onToggleExpand,
  onRemove,
  onRunCombo,
  onOpenSetupWizard,
}) {
  const applyState = combo?.applyState || 'ready';
  const isReady = applyState === 'ready';
  const isPendingImpl = applyState === 'pending_implementation';
  const isPendingCfg = applyState === 'pending_config';
  const isPendingFlagMap = applyState === 'pending_flag_map';
  const pendingHs = combo?.pendingHypotheses || [];
  const bareList = combo?.bareBindings || [];
  const ginPath = combo?.configPathProposed || combo?.config?.gin_config || '';
  const readyVia = combo?.readyVia || '';
  const resolvedBindings = combo?.resolvedBindings || [];

  const tooltipParts = [];
  if (isPendingImpl) {
    tooltipParts.push(
      `Pending implementation of: ${pendingHs.join(', ') || '...'}. `
      + 'Will become runnable when these hypotheses are implemented.'
    );
  }
  if (isPendingCfg) {
    tooltipParts.push(
      `Pending config: ${ginPath || combo?.configStatus || 'needs_generation'}. `
      + 'Will become runnable when the gin config is generated.'
    );
  }
  if (isPendingFlagMap) {
    tooltipParts.push(
      `Hub supports flag composition, but ${bareList.join(', ') || '...'} `
      + 'lack scoped bindings (e.g., hstu_encoder.enable_h1). '
      + 'Open the SetupWizard to fix the per-hub hypothesisFlagMap. '
      + 'Submit would otherwise be rejected at runtime.'
    );
  }
  const tooltip = tooltipParts.join(' ');

  return (
    <Card
      variant="outlined"
      sx={{
        opacity: isReady ? 1 : 0.65,
        bgcolor: isReady ? 'transparent' : 'rgba(255,255,255,0.02)',
      }}
    >
      <CardContent sx={{ p: 1, '&:last-child': { pb: 1 } }}>
        <Box
          sx={{
            display: 'flex', alignItems: 'center', gap: 1,
            flexWrap: 'wrap', cursor: 'pointer',
          }}
          onClick={onToggleExpand}
          role="button"
          aria-expanded={!!expanded}
        >
          {onToggleCheck && (
            <Checkbox
              size="small"
              checked={!!checked}
              onChange={(e) => { e.stopPropagation(); onToggleCheck(); }}
              onClick={(e) => e.stopPropagation()}
              sx={{ p: 0.5 }}
            />
          )}
          <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
            {combo.comboId || combo.futureComboId || '?'}
          </Typography>
          {combo.comboKey && (
            <Chip
              label={combo.comboKey}
              size="small"
              sx={{ fontSize: '0.62rem', height: 20, fontFamily: 'monospace' }}
            />
          )}
          {(combo.selectedItems || []).map(h => (
            <Chip
              key={h}
              label={h}
              size="small"
              variant="outlined"
              sx={{ fontSize: '0.62rem', height: 18 }}
            />
          ))}
          <Box sx={{ flex: 1 }} />
          {isPendingImpl && (
            <Tooltip title={tooltip} placement="top" arrow>
              <Chip
                icon={<HourglassEmptyIcon fontSize="inherit" />}
                label={`Pending impl: ${pendingHs.slice(0, 3).join(', ')}${pendingHs.length > 3 ? `, +${pendingHs.length - 3}` : ''}`}
                size="small"
                color="warning"
                variant="outlined"
                sx={{ fontSize: '0.62rem', height: 22 }}
              />
            </Tooltip>
          )}
          {isPendingCfg && (
            <Tooltip title={tooltip} placement="top" arrow>
              <Chip
                icon={<BuildIcon fontSize="inherit" />}
                label={`Pending config: ${ginPath || combo?.configStatus || 'needs_generation'}`}
                size="small"
                color="warning"
                variant="outlined"
                sx={{ fontSize: '0.62rem', height: 22, maxWidth: 320, '& .MuiChip-label': { overflow: 'hidden', textOverflow: 'ellipsis' } }}
              />
            </Tooltip>
          )}
          {isPendingFlagMap && (
            <Tooltip title={tooltip} placement="top" arrow>
              <Chip
                icon={<VpnKeyOffIcon fontSize="inherit" />}
                label={`Pending flag map: ${bareList.slice(0, 3).join(', ')}${bareList.length > 3 ? `, +${bareList.length - 3}` : ''}`}
                size="small"
                color="warning"
                variant="outlined"
                onClick={
                  onOpenSetupWizard
                    ? (e) => { e.stopPropagation(); onOpenSetupWizard(); }
                    : undefined
                }
                sx={{
                  fontSize: '0.62rem', height: 22,
                  maxWidth: 320, '& .MuiChip-label': { overflow: 'hidden', textOverflow: 'ellipsis' },
                  cursor: onOpenSetupWizard ? 'pointer' : 'default',
                }}
              />
            </Tooltip>
          )}
          <Stack direction="row" spacing={0.25} onClick={(e) => e.stopPropagation()}>
            {onRemove && (
              <Tooltip title="Remove combo from active set">
                <IconButton size="small" onClick={() => onRemove()} sx={{ p: 0.5 }}>
                  <DeleteOutlineIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            )}
            <IconButton size="small" onClick={onToggleExpand} sx={{ p: 0.5 }}>
              {expanded ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
            </IconButton>
          </Stack>
        </Box>
        <Collapse in={!!expanded} unmountOnExit>
          <Box sx={{ pt: 1, pl: 4 }}>
            {combo.title && (
              <Typography variant="body2" sx={{ fontSize: '0.78rem', mb: 0.5 }}>
                {combo.title}
              </Typography>
            )}
            {combo.rationale && (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', fontSize: '0.72rem' }}>
                {combo.rationale}
              </Typography>
            )}
            {readyVia === 'flag_composition' && resolvedBindings.length > 0 ? (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', fontSize: '0.7rem', fontFamily: 'monospace', mt: 0.5 }}>
                flags: {resolvedBindings.join(', ')}
              </Typography>
            ) : ginPath ? (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', fontSize: '0.7rem', fontFamily: 'monospace', mt: 0.5 }}>
                gin: {ginPath}
              </Typography>
            ) : null}
            {onRunCombo && (
              <Box sx={{ mt: 1, display: 'flex', justifyContent: 'flex-end' }}>
                <Tooltip
                  title={!isReady
                    ? `Disabled — ${
                      isPendingImpl ? 'pending implementation'
                        : isPendingFlagMap ? 'pending flag map'
                          : 'pending config'
                    }. ${tooltip}`
                    : ''}
                  placement="top"
                  arrow
                  disableHoverListener={isReady}
                >
                  <span>
                    <Button
                      size="small"
                      variant="outlined"
                      disabled={!isReady}
                      onClick={() => onRunCombo(combo)}
                      sx={{ textTransform: 'none', fontSize: '0.72rem' }}
                    >
                      Run
                    </Button>
                  </span>
                </Tooltip>
              </Box>
            )}
          </Box>
        </Collapse>
      </CardContent>
    </Card>
  );
}

export default function ComboReviewSection({
  activeCombos,
  selectedComboIds,
  expandedComboIds,
  onToggleCombo,
  onToggleExpand,
  onRemoveCombo,
  onSelectAll,
  onClearSelected,
  onExpandAll,
  onCollapseAll,
  onRunCombo,
  onOpenSetupWizard,
}) {
  if (!Array.isArray(activeCombos) || activeCombos.length === 0) {
    return null;
  }

  const ready = activeCombos.filter(c => (c?.applyState || 'ready') === 'ready');
  const pending = activeCombos.filter(c => (c?.applyState || 'ready') !== 'ready');
  const ordered = [...ready, ...pending];
  const allIds = ordered.map(c => c?.comboId).filter(Boolean);
  const selectedCount = (selectedComboIds && selectedComboIds.size) || 0;
  const expandedCount = (expandedComboIds && expandedComboIds.size) || 0;
  const allSelected = selectedCount === allIds.length && allIds.length > 0;
  const allExpanded = expandedCount === allIds.length && allIds.length > 0;

  return (
    <Box sx={{ px: 2, pt: 1.5 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1, gap: 1, flexWrap: 'wrap' }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 600, fontSize: '0.85rem' }}>
          Active &amp; Pending Combos
        </Typography>
        <Typography variant="caption" color="text.secondary">
          ({ready.length} active · {pending.length} pending · {selectedCount} checked)
        </Typography>
        <Box sx={{ flex: 1 }} />
        {(onExpandAll || onCollapseAll) && (
          <Button
            size="small"
            onClick={() => {
              if (allExpanded) onCollapseAll && onCollapseAll();
              else onExpandAll && onExpandAll();
            }}
            sx={{ textTransform: 'none', fontSize: '0.72rem' }}
          >
            {allExpanded ? 'Collapse all' : 'Expand all'}
          </Button>
        )}
        {(onSelectAll || onClearSelected) && (
          <Button
            size="small"
            onClick={() => {
              if (allSelected) onClearSelected && onClearSelected();
              else onSelectAll && onSelectAll();
            }}
            sx={{ textTransform: 'none', fontSize: '0.72rem' }}
          >
            {allSelected ? 'Clear' : 'Select all'}
          </Button>
        )}
      </Box>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
        {ordered.map(combo => {
          const id = combo?.comboId || combo?.futureComboId || combo?.comboKey;
          return (
            <ComboCard
              key={id}
              combo={combo}
              checked={selectedComboIds ? selectedComboIds.has(combo?.comboId) : undefined}
              expanded={expandedComboIds ? expandedComboIds.has(combo?.comboId) : false}
              onToggleCheck={onToggleCombo ? () => onToggleCombo(combo?.comboId) : undefined}
              onToggleExpand={() => onToggleExpand && onToggleExpand(combo?.comboId)}
              onRemove={onRemoveCombo ? () => onRemoveCombo(combo?.comboId) : undefined}
              onRunCombo={onRunCombo}
              onOpenSetupWizard={onOpenSetupWizard}
            />
          );
        })}
      </Box>
    </Box>
  );
}

/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * SubmissionActionButton — single stateful primary button that drives the
 * Experiment Hub's submission flow (label/icon/color/onClick swap on
 * `status`). The ⚙ split-icon opens secondary actions.
 *
 * Ported verbatim from RankEvolve (components/hub/SubmissionActionButton.js).
 */

import React, { useCallback, useMemo, useState } from 'react';
import {
  Box,
  Button,
  ButtonGroup,
  IconButton,
  Menu,
  MenuItem,
  Snackbar,
  Tooltip,
} from '@mui/material';
import SettingsIcon from '@mui/icons-material/Settings';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import RefreshIcon from '@mui/icons-material/Refresh';
import HourglassTopIcon from '@mui/icons-material/HourglassTop';
import VisibilityIcon from '@mui/icons-material/Visibility';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';

const STATUS_NOT_STARTED = 'not_started';
const STATUS_IN_PROGRESS = 'in_progress';
const STATUS_READY = 'ready';
const STATUS_ERROR = 'error';

export default function SubmissionActionButton({
  status,
  selectedIds,
  setup,
  onSetupClick,
  onRegenerateClick,
  onSubmitClick,
  onViewScriptClick,
  onJumpToTaskClick,
  onViewInputsClick,
  onViewErrorClick,
  externalDisabled = false,
  externalDisabledReason = '',
  selectedCombos = null,
}) {
  const [menuAnchor, setMenuAnchor] = useState(null);
  const [toast, setToast] = useState({ open: false, message: '' });

  const comboBatch = Array.isArray(selectedCombos)
    ? selectedCombos.filter(Boolean)
    : [];
  const hasSelection =
    comboBatch.length > 0 || !!(selectedIds && selectedIds.size > 0);
  const showSecondaryMenu = status === STATUS_READY || status === STATUS_ERROR;

  const config = useMemo(() => {
    if (status === STATUS_IN_PROGRESS) {
      return {
        label: 'Creating runner — click to view progress',
        icon: <HourglassTopIcon />,
        color: 'info',
        variant: 'outlined',
        disabled: false,
        onClick: () => {
          if (!setup?.taskId) {
            setToast({
              open: true,
              message: 'Task starting — try again in a moment',
            });
            return;
          }
          onJumpToTaskClick && onJumpToTaskClick(setup.taskId);
        },
      };
    }
    if (status === STATUS_READY) {
      let baseLabel;
      if (comboBatch.length > 0) {
        if (comboBatch.length <= 3) {
          const ids = comboBatch.map(c => c?.comboId || '?').join(', ');
          baseLabel = `Submit ${comboBatch.length} Experiment${comboBatch.length === 1 ? '' : 's'} (${ids})`;
        } else {
          baseLabel = `Submit ${comboBatch.length} Experiments`;
        }
      } else {
        const selectionLabel = hasSelection
          ? ` (${[...(selectedIds || [])].join(', ')})`
          : ' (select hypotheses first)';
        baseLabel = `Submit Experiment${selectionLabel}`;
      }
      const blocked = !hasSelection || externalDisabled;
      const finalLabel = externalDisabled && externalDisabledReason
        ? `${baseLabel} — ${externalDisabledReason}`
        : baseLabel;
      return {
        label: finalLabel,
        icon: <RocketLaunchIcon />,
        color: 'primary',
        variant: 'contained',
        disabled: blocked,
        tooltip: externalDisabled && externalDisabledReason ? externalDisabledReason : '',
        onClick: blocked ? undefined : onSubmitClick,
      };
    }
    if (status === STATUS_ERROR) {
      return {
        label: 'Retry',
        icon: <RefreshIcon />,
        color: 'warning',
        variant: 'contained',
        disabled: false,
        onClick: onRegenerateClick,
      };
    }
    // STATUS_NOT_STARTED (default)
    return {
      label: 'Create Experiment Runner',
      icon: <SettingsIcon />,
      color: 'primary',
      variant: 'outlined',
      disabled: false,
      onClick: onSetupClick,
    };
  }, [
    status,
    hasSelection,
    selectedIds,
    comboBatch,
    setup?.taskId,
    onSetupClick,
    onRegenerateClick,
    onSubmitClick,
    onJumpToTaskClick,
    externalDisabled,
    externalDisabledReason,
  ]);

  const handleMenuOpen = useCallback((event) => {
    setMenuAnchor(event.currentTarget);
  }, []);
  const handleMenuClose = useCallback(() => setMenuAnchor(null), []);

  const handleMenuItem = useCallback(
    (handler) => () => {
      setMenuAnchor(null);
      if (handler) handler();
    },
    [],
  );

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
      {showSecondaryMenu ? (
        <ButtonGroup
          variant={config.variant}
          color={config.color}
          fullWidth
          aria-label="Submission actions"
        >
          <Button
            startIcon={config.icon}
            onClick={config.onClick}
            disabled={config.disabled}
            sx={{
              textTransform: 'none',
              fontSize: '0.9rem',
              fontWeight: 600,
              flex: 1,
              justifyContent: 'flex-start',
            }}
            aria-label={config.label}
          >
            {config.label}
          </Button>
          <Tooltip title="More setup actions">
            <IconButton
              onClick={handleMenuOpen}
              aria-label="More setup actions"
              size="small"
              sx={{ borderLeft: '1px solid', borderColor: 'divider', borderRadius: 0, px: 1.5 }}
            >
              <SettingsIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </ButtonGroup>
      ) : (
        <Button
          fullWidth
          variant={config.variant}
          color={config.color}
          startIcon={config.icon}
          onClick={config.onClick}
          disabled={config.disabled}
          sx={{
            textTransform: 'none',
            fontSize: '0.9rem',
            fontWeight: 600,
            justifyContent: 'flex-start',
          }}
          aria-label={config.label}
        >
          {config.label}
        </Button>
      )}

      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={handleMenuClose}
        anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
        transformOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        {status === STATUS_READY && onViewScriptClick && (
          <MenuItem onClick={handleMenuItem(onViewScriptClick)}>
            <VisibilityIcon fontSize="small" sx={{ mr: 1 }} />
            View / edit runner
          </MenuItem>
        )}
        {(status === STATUS_READY || status === STATUS_ERROR)
          && onRegenerateClick
          && status !== STATUS_ERROR /* Retry main button already does this */ && (
          <MenuItem onClick={handleMenuItem(onRegenerateClick)}>
            <RefreshIcon fontSize="small" sx={{ mr: 1 }} />
            Re-create runner
          </MenuItem>
        )}
        {onViewInputsClick && (
          <MenuItem onClick={handleMenuItem(onViewInputsClick)}>
            <SettingsIcon fontSize="small" sx={{ mr: 1 }} />
            View setup inputs
          </MenuItem>
        )}
        {status === STATUS_ERROR && onViewErrorClick && (
          <MenuItem onClick={handleMenuItem(onViewErrorClick)}>
            <ErrorOutlineIcon fontSize="small" sx={{ mr: 1 }} />
            View error details
          </MenuItem>
        )}
      </Menu>

      <Snackbar
        open={toast.open}
        autoHideDuration={2500}
        onClose={() => setToast({ open: false, message: '' })}
        message={toast.message}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      />
    </Box>
  );
}

/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * SplitActionButton — reusable primary + secondary control with
 * auto-degradation by `secondary` length:
 *
 *   secondary = []  →  plain <Button>
 *   secondary = [oneDescriptor] OR a single descriptor object
 *                  →  inline split (<ButtonGroup> with two buttons; thin divider)
 *   secondary = [twoOrMore descriptors]
 *                  →  split + dropdown menu (▾ trigger toggles MUI <Menu>)
 *
 * Plan: humming-tinkering-wirth.md v5 §B.
 *
 * Justification (rule of two): SubmissionActionButton.js:172-204 already
 * paints this primary+secondary pattern by hand. The new Accumulated-
 * Learnings refresh is the second adopter — extracting now stops both
 * call sites from drifting on the hand-painted divider.
 */

import React, { forwardRef, useRef, useState, useCallback } from 'react';
import {
  Badge,
  Button,
  ButtonGroup,
  CircularProgress,
  ListItemIcon,
  Menu,
  MenuItem,
  Tooltip,
  Typography,
} from '@mui/material';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';

/**
 * Wrap a Button in a Tooltip with the MUI footgun fix: <Tooltip> over a
 * disabled MUI <Button> renders nothing without an inline-flex <span>.
 */
function _wrap(child, tip, disabledTip, isDisabled) {
  const title = (isDisabled ? (disabledTip ?? tip) : tip) ?? '';
  if (!title) return child;
  return (
    <Tooltip title={title}>
      <span style={{ display: 'inline-flex' }}>{child}</span>
    </Tooltip>
  );
}

/**
 * Normalize secondary into an array. Accepts:
 *  - undefined → []
 *  - one descriptor object → [object]
 *  - array → array (passed through)
 */
function _normalizeSecondary(secondary) {
  if (secondary == null) return [];
  if (Array.isArray(secondary)) return secondary;
  return [secondary];
}

const SplitActionButton = forwardRef(function SplitActionButton(props, ref) {
  const {
    variant = 'outlined',
    color = 'primary',
    size = 'small',
    fullWidth = false,
    secondaryWidth = 32,
    sx,
    primary,
    secondary,
    secondaryRef,         // forwarded to the secondary segment OR the dropdown trigger
    secondaryBadge,
  } = props;

  const items = _normalizeSecondary(secondary);
  const spinnerSize = size === 'small' ? 14 : size === 'large' ? 22 : 18;

  // ── Mode 0 — no secondary: degrade to plain <Button>.
  if (items.length === 0) {
    const btn = (
      <Button
        ref={ref}
        variant={variant}
        color={color}
        size={size}
        fullWidth={fullWidth}
        startIcon={primary?.icon}
        onClick={primary?.onClick}
        disabled={!!primary?.disabled || !!primary?.loading}
        sx={{ textTransform: 'none', fontSize: '0.78rem', ...(sx || {}) }}
      >
        {primary?.loading ? <CircularProgress size={spinnerSize} /> : primary?.label}
      </Button>
    );
    return _wrap(btn, primary?.tooltip, primary?.disabledTooltip, !!primary?.disabled);
  }

  // ── Primary segment (shared by inline-split and dropdown modes).
  const primaryBtn = (
    <Button
      onClick={primary?.onClick}
      disabled={!!primary?.disabled || !!primary?.loading}
      startIcon={primary?.loading ? <CircularProgress size={spinnerSize} /> : primary?.icon}
      sx={{ textTransform: 'none', fontSize: '0.78rem' }}
    >
      {primary?.label}
    </Button>
  );

  // ── Mode 1 — single secondary: inline split.
  if (items.length === 1) {
    const sec = items[0];
    let secBtn = (
      <Button
        ref={secondaryRef}
        onClick={sec.onClick}
        disabled={!!sec.disabled || !!sec.loading}
        aria-label={sec.ariaLabel || (typeof sec.tooltip === 'string' ? sec.tooltip : 'secondary action')}
        sx={{
          minWidth: secondaryWidth, px: 1,
          borderLeft: '1px solid',
          borderColor: 'divider',
          borderRadius: 0,
        }}
      >
        {sec.loading ? <CircularProgress size={spinnerSize} /> : sec.icon}
      </Button>
    );
    if (secondaryBadge) {
      secBtn = (
        <Badge
          badgeContent={secondaryBadge.content}
          color={secondaryBadge.color || 'error'}
          variant={secondaryBadge.variant || 'dot'}
        >
          {secBtn}
        </Badge>
      );
    }
    return (
      <ButtonGroup
        ref={ref}
        variant={variant} color={color} size={size}
        fullWidth={fullWidth} sx={sx}
      >
        {_wrap(primaryBtn, primary?.tooltip, primary?.disabledTooltip, !!primary?.disabled)}
        {_wrap(secBtn, sec.tooltip, sec.disabledTooltip, !!sec.disabled)}
      </ButtonGroup>
    );
  }

  // ── Mode N — split + dropdown menu.
  return (
    <DropdownVariant
      ref={ref}
      variant={variant} color={color} size={size}
      fullWidth={fullWidth} sx={sx}
      secondaryWidth={secondaryWidth}
      primary={primary}
      primaryBtn={primaryBtn}
      items={items}
      spinnerSize={spinnerSize}
      secondaryRef={secondaryRef}
      secondaryBadge={secondaryBadge}
    />
  );
});

const DropdownVariant = forwardRef(function DropdownVariant({
  variant, color, size, fullWidth, sx, secondaryWidth,
  primary, primaryBtn, items, spinnerSize, secondaryRef, secondaryBadge,
}, ref) {
  const localAnchorRef = useRef(null);
  const anchorRef = secondaryRef || localAnchorRef;
  const [open, setOpen] = useState(false);

  const handleToggle = useCallback(() => setOpen((o) => !o), []);
  const handleClose = useCallback(() => setOpen(false), []);

  const anyLoading = items.some((it) => it.loading);
  const triggerAriaLabel = `More actions for ${primary?.label || 'this control'}`;

  let trigger = (
    <Button
      ref={anchorRef}
      onClick={handleToggle}
      aria-haspopup="menu"
      aria-expanded={open ? 'true' : 'false'}
      aria-label={triggerAriaLabel}
      sx={{
        minWidth: secondaryWidth, px: 0.5,
        borderLeft: '1px solid',
        borderColor: 'divider',
        borderRadius: 0,
      }}
    >
      <ArrowDropDownIcon fontSize="small" />
    </Button>
  );
  if (anyLoading || secondaryBadge) {
    const badgeProps = secondaryBadge
      ? {
          badgeContent: secondaryBadge.content,
          color: secondaryBadge.color || 'error',
          variant: secondaryBadge.variant || 'dot',
        }
      : { variant: 'dot', color: 'error', invisible: false };
    trigger = (
      <Badge {...badgeProps}>
        {trigger}
      </Badge>
    );
  }

  return (
    <>
      <ButtonGroup
        ref={ref}
        variant={variant} color={color} size={size}
        fullWidth={fullWidth} sx={sx}
      >
        {_wrap(primaryBtn, primary?.tooltip, primary?.disabledTooltip, !!primary?.disabled)}
        {trigger}
      </ButtonGroup>
      <Menu
        anchorEl={anchorRef.current}
        open={open}
        onClose={handleClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
        MenuListProps={{ dense: true }}
      >
        {items.map((it) => {
          const isDisabled = !!it.disabled || !!it.loading;
          const item = (
            <MenuItem
              key={it.key}
              disabled={isDisabled}
              onClick={() => {
                handleClose();
                if (!isDisabled && typeof it.onClick === 'function') it.onClick();
              }}
              aria-label={it.ariaLabel || it.label}
            >
              <ListItemIcon>
                {it.loading ? <CircularProgress size={spinnerSize} /> : it.icon}
              </ListItemIcon>
              <Typography variant="body2" sx={{ fontSize: '0.85rem' }}>
                {it.label}
              </Typography>
            </MenuItem>
          );
          // MUI tooltip-on-disabled-MenuItem also needs the <span> dance.
          const title = (isDisabled ? (it.disabledTooltip ?? it.tooltip) : it.tooltip) ?? '';
          if (!title) return item;
          return (
            <Tooltip key={it.key} title={title} placement="left">
              <span style={{ display: 'block' }}>{item}</span>
            </Tooltip>
          );
        })}
      </Menu>
    </>
  );
});

export default SplitActionButton;

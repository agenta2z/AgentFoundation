/**
 * EllipsisText — single-line text that truncates with an ellipsis, and reports
 * whether it is *actually* truncated (overflowing its container).
 *
 * Used where a duplicate "full text" view should appear ONLY when the inline
 * text doesn't fit on one line (e.g. a select's option description vs. a
 * separate Details box). Re-measures on text change and on container resize.
 *
 * Props:
 *   text                - the string to render
 *   onTruncatedChange   - (bool) => void, fired only when truncation flips
 *   variant             - MUI Typography variant (default "caption")
 *   sx                  - extra styles merged onto the one-line container
 */

import React, { useRef, useState, useCallback, useLayoutEffect, useEffect } from 'react';
import { Typography } from '@mui/material';

export default function EllipsisText({ text, onTruncatedChange, variant = 'caption', sx, ...rest }) {
  const ref = useRef(null);
  const lastReported = useRef(null);

  const measure = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    // +1 guards against sub-pixel rounding producing false positives.
    const truncated = el.scrollWidth > el.clientWidth + 1;
    if (truncated !== lastReported.current) {
      lastReported.current = truncated;
      onTruncatedChange?.(truncated);
    }
  }, [onTruncatedChange]);

  // Measure before paint so the dependent UI (e.g. a Details box) doesn't flash.
  useLayoutEffect(() => { measure(); }, [text, measure]);

  // Re-measure when the container resizes (window/layout changes).
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(() => measure());
    ro.observe(el);
    return () => ro.disconnect();
  }, [measure]);

  return (
    <Typography
      ref={ref}
      variant={variant}
      sx={{ display: 'block', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', ...sx }}
      {...rest}
    >
      {text}
    </Typography>
  );
}

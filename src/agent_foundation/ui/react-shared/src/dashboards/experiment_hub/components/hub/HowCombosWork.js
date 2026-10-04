/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * HowCombosWork — collapsible info panel explaining how hypothesis combos
 * compose via scoped feature flags + a per-attempt gin overlay.
 *
 * Ported verbatim from RankEvolve (components/hub/HowCombosWork.js).
 */

import React, { useState } from 'react';
import {
  Box,
  Collapse,
  IconButton,
  Typography,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';

export default function HowCombosWork({ defaultExpanded = false }) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  return (
    <Box
      sx={{
        border: '1px solid',
        borderColor: 'rgba(255,255,255,0.08)',
        borderRadius: 1,
        backgroundColor: 'rgba(74, 144, 217, 0.04)',
        mb: 1,
      }}
    >
      <Box
        sx={{
          display: 'flex', alignItems: 'center', cursor: 'pointer',
          px: 1.5, py: 0.5,
        }}
        onClick={() => setExpanded((e) => !e)}
      >
        <HelpOutlineIcon sx={{ fontSize: '0.9rem', mr: 0.75, color: 'primary.main' }} />
        <Typography variant="caption" sx={{ flexGrow: 1, fontSize: '0.74rem', color: 'text.primary' }}>
          How Combos Work — flag-based composition (no new gin file per combo)
        </Typography>
        <IconButton size="small" sx={{ p: 0.25 }}>
          {expanded ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
        </IconButton>
      </Box>
      <Collapse in={expanded}>
        <Box sx={{ px: 2, pb: 1.5, fontSize: '0.72rem', '& code': { fontFamily: 'monospace', fontSize: '0.7rem', px: 0.5, backgroundColor: 'rgba(255,255,255,0.06)', borderRadius: 0.5 } }}>
          <Box component="ol" sx={{ pl: 2, m: 0, '& li': { mb: 0.5, lineHeight: 1.5 } }}>
            <li>
              Each implemented hypothesis = one <code>enable_&lt;name&gt;: bool = False</code> field
              on the model's <code>@gin.configurable</code> config class
              (mandated by <code>hypothesis_implementation/default.jinja2</code>).
              When the flag is False, the model is byte-for-byte identical to the original.
            </li>
            <li>
              Combos are <strong>subsets of flags to flip</strong>. The runner composes
              a per-attempt gin overlay at submit time. <strong>Never a new file per combo.</strong>
            </li>
            <li>
              <strong>Flag names are SCOPED</strong> (e.g., <code>hstu_encoder.enable_h17</code>),
              so they bind to the right <code>@gin.configurable</code> class.
              Bare names like <code>enable_h17</code> are top-level gin macros and
              <strong> do NOT bind to model fields</strong> — the runner rejects bare
              names at submit time to prevent silent no-op runs.
            </li>
            <li>
              <strong>Local HSTU runs</strong>: per-attempt overlay file at
              <code>&lt;workspace&gt;/_monitor/overlay_attempt_&lt;N&gt;.gin</code>
              with <code>include 'baseline.gin'</code> + one
              <code>&lt;scope&gt;.&lt;flag&gt; = True</code> line per enabled hypothesis.
              <strong> FBLearner</strong>: <code>--enable-flags &lt;scope1&gt;.&lt;flag1&gt;,&lt;scope2&gt;.&lt;flag2&gt;</code>
              CLI passed to <code>submit_v&lt;n&gt;.py</code>.
            </li>
            <li>
              <strong>Pre-flight grep</strong>
              (<code>experiment_combos_bridge.preflight_check_flags</code>) validates
              all <code>enable_&lt;id&gt;</code> declarations exist in source before
              launch. Combos with missing flags fail loudly, not silently.
            </li>
          </Box>
          <Typography variant="caption" sx={{ display: 'block', mt: 1, fontSize: '0.66rem', color: 'text.secondary' }}>
            Need ad-hoc parameter tuning (e.g., a non-default
            <code> input_compression_budget</code>)? Pass inline gin bindings
            in <code>--enable-flags</code>, e.g.,
            <code> 'hstu_encoder.enable_h17,train_fn.input_compression_budget=300'</code>.
          </Typography>
        </Box>
      </Collapse>
    </Box>
  );
}

/**
 * Breadcrumb — clickable hierarchy navigation for drill-down graph views.
 *
 * Shows: Pipeline > Worker 2: Research... > [current level]
 * Each segment is clickable to navigate back to that depth.
 *
 * Props:
 *   graphPath     - array of node IDs representing current drill-down path
 *   task          - task object (to look up node labels from parent graphs)
 *   onNavigate(depth) - called with new graphPath length to navigate to
 */

import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';

function getBreadcrumbLabels(task, graphPath) {
  const labels = ['Pipeline'];
  let graph = task?.graph;
  for (let i = 0; i < graphPath.length; i++) {
    const nodeId = graphPath[i];
    const node = graph?.nodes?.find(n => n.id === nodeId);
    labels.push(node?.label || nodeId);
    const key = graphPath.slice(0, i + 1).join('/');
    graph = task?.subGraphs?.[key] || null;
  }
  return labels;
}

export function Breadcrumb({ graphPath, task, onNavigate }) {
  if (!graphPath || graphPath.length === 0) return null;

  const labels = getBreadcrumbLabels(task, graphPath);

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        px: 2,
        py: 0.5,
        backgroundColor: 'rgba(255,255,255,0.03)',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
        gap: 0.5,
        flexShrink: 0,
        minHeight: 28,
      }}
    >
      {labels.map((label, i) => {
        const isLast = i === labels.length - 1;
        const isCurrent = i === graphPath.length;
        return (
          <React.Fragment key={i}>
            {i > 0 && (
              <Typography variant="caption" sx={{ color: 'text.disabled', mx: 0.25 }}>
                ›
              </Typography>
            )}
            <Typography
              variant="caption"
              onClick={() => !isLast && onNavigate(i)}
              sx={{
                fontSize: '0.68rem',
                color: isLast ? 'text.primary' : 'primary.main',
                cursor: isLast ? 'default' : 'pointer',
                fontWeight: isLast ? 600 : 400,
                '&:hover': isLast ? {} : { textDecoration: 'underline' },
                maxWidth: 180,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
              title={label}
            >
              {label}
            </Typography>
          </React.Fragment>
        );
      })}
      <Box sx={{ flex: 1 }} />
      <IconButton
        size="small"
        onClick={() => onNavigate(graphPath.length - 1)}
        sx={{ color: 'text.disabled', p: 0.25 }}
        title="Back (Esc)"
      >
        <CloseIcon sx={{ fontSize: 14 }} />
      </IconButton>
    </Box>
  );
}

export default Breadcrumb;

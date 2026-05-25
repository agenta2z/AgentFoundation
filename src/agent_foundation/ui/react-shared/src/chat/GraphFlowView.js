/**
 * GraphFlowView — horizontal flow graph with Focus+Context zoom.
 *
 * Two view modes:
 *   page-switch:    renders only the current sub-graph (original behavior)
 *   focus-context:  renders ALL nodes in a unified canvas; clicking a container
 *                   zooms the viewport to its sub-graph while outer nodes fade
 *                   to ghost outlines (§3.5.1-3.5.6 of the plan)
 *
 * Props (page-switch mode):
 *   nodes, edges, selectedNodeId, expandableNodeIds, onNodeClick
 *
 * Props (focus-context mode — additional):
 *   viewMode        'page-switch' | 'focus-context'
 *   subGraphs       { parentNodeId: { nodes, edges } }
 *   focusedPath     string[] — current drill-down path
 *   onFocusChange   (path) => void — called when user clicks container or ghost
 */

import React, { useMemo, useState, useRef, useCallback, useEffect } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import CircularProgress from '@mui/material/CircularProgress';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import ZoomOutIcon from '@mui/icons-material/ZoomOut';
import FitScreenIcon from '@mui/icons-material/FitScreen';
import IconButton from '@mui/material/IconButton';
import Tooltip from '@mui/material/Tooltip';
import { useTheme } from '@mui/material/styles';

const NORMAL = { nodeH: 82, nodeW: 160, rowGap: 10 };
const COMPACT = { nodeH: 42, nodeW: 140, rowGap: 6 };
const SUB_SCALE = 0.6;
const COL_GAP = 60;
const SUB_GAP = 40;
const PADDING = 16;
const MIN_ZOOM = 0.15;
const MAX_ZOOM = 3.0;
const ZOOM_STEP = 0.15;
const GHOST_SIZE = 32;
const GROUP_THRESHOLD = 6;
const DOTS_PER_ROW = 6;
const DOT_SIZE = 24;
const DOT_STRIDE = DOT_SIZE + 2;
const MINI_CARD_H = 30;
const GROUP_HEADER_H = 26;

const STATUS_CONFIG = {
  pending:   { color: '#666', chipColor: 'default',  icon: null },
  running:   { color: '#1976d2', chipColor: 'info',  icon: 'spinner' },
  completed: { color: '#2e7d32', chipColor: 'success', icon: 'check' },
  error:     { color: '#c62828', chipColor: 'error', icon: 'error' },
  skipped:   { color: '#f57c00', chipColor: 'warning', icon: 'check' },
};

function computeLayout(nodes, edges, dims) {
  if (!nodes || !nodes.length) return { columns: [], totalW: 0, totalH: 0, edgePaths: [], positions: {}, groupedColumns: {} };
  const { nodeH, nodeW, rowGap } = dims;
  const outEdges = {};
  const inDegree = {};
  nodes.forEach(n => { outEdges[n.id] = []; inDegree[n.id] = 0; });
  (edges || []).forEach(e => {
    if (outEdges[e.source]) outEdges[e.source].push(e.target);
    if (e.target in inDegree) inDegree[e.target] = (inDegree[e.target] || 0) + 1;
  });
  const depth = {};
  const queue = nodes.filter(n => (inDegree[n.id] || 0) === 0).map(n => n.id);
  queue.forEach(id => { depth[id] = 0; });
  const visited = new Set(queue);
  let head = 0;
  while (head < queue.length) {
    const cur = queue[head++];
    (outEdges[cur] || []).forEach(child => {
      const d = (depth[cur] || 0) + 1;
      if (depth[child] === undefined || depth[child] < d) depth[child] = d;
      if (!visited.has(child)) { visited.add(child); queue.push(child); }
    });
  }
  nodes.forEach(n => { if (depth[n.id] === undefined) depth[n.id] = 0; });
  const maxDepth = Math.max(...Object.values(depth));
  const columns = Array.from({ length: maxDepth + 1 }, () => []);
  nodes.forEach(n => columns[depth[n.id] || 0].push(n));

  // Detect grouped columns BEFORE computing positions so their actual width
  // feeds into column spacing (prevents overlap when groupW > nodeW).
  const groupedColumns = {};
  columns.forEach((col, ci) => {
    if (col.length > GROUP_THRESHOLD) {
      const running = col.filter(n => n.status === 'running');
      const others = col.filter(n => n.status !== 'running');
      const runH = running.length * MINI_CARD_H;
      const dotRows = Math.ceil(others.length / DOTS_PER_ROW);
      const dotsH = dotRows * DOT_STRIDE + 8;
      const groupH = GROUP_HEADER_H + runH + dotsH + 12;
      const groupW = Math.max(nodeW, DOTS_PER_ROW * DOT_STRIDE + 12);
      groupedColumns[ci] = {
        y: PADDING, w: groupW, h: groupH,
        nodeIds: new Set(col.map(n => n.id)),
        nodes: col,
      };
    }
  });

  // Adaptive column x-offsets: use actual column width (groupW or nodeW)
  const colX = [];
  let curX = PADDING;
  for (let ci = 0; ci < columns.length; ci++) {
    colX[ci] = curX;
    curX += (groupedColumns[ci] ? groupedColumns[ci].w : nodeW) + COL_GAP;
  }

  // Assign positions using adaptive x-offsets
  const positions = {};
  columns.forEach((col, ci) => {
    col.forEach((n, ri) => {
      positions[n.id] = {
        x: colX[ci],
        y: PADDING + ri * (nodeH + rowGap),
        cx: colX[ci] + nodeW / 2,
        cy: PADDING + ri * (nodeH + rowGap) + nodeH / 2,
      };
    });
  });

  // Set grouped column x from adaptive offsets
  for (const ci of Object.keys(groupedColumns)) {
    groupedColumns[ci].x = colX[ci];
  }

  const totalW = curX - COL_GAP + PADDING;
  const edgePaths = (edges || []).map(e => {
    const src = positions[e.source];
    const tgt = positions[e.target];
    if (!src || !tgt) return null;
    const sx = src.x + nodeW, sy = src.cy, tx = tgt.x, ty = tgt.cy;
    const cp = (tx - sx) * 0.5;
    return { key: `${e.source}-${e.target}`, d: `M ${sx} ${sy} C ${sx + cp} ${sy}, ${tx - cp} ${ty}, ${tx} ${ty}` };
  }).filter(Boolean);
  const totalH = PADDING * 2 + Math.max(...columns.map((col, ci) =>
    groupedColumns[ci] ? groupedColumns[ci].h : (col.length * nodeH + Math.max(0, col.length - 1) * rowGap)
  ));
  return { columns, positions, totalW, totalH, edgePaths, groupedColumns };
}

/**
 * Compute unified layout: root graph + all sub-graphs in one coordinate space.
 * Sub-graphs are positioned to the right of their parent container node.
 */
function computeUnifiedLayout(rootNodes, rootEdges, subGraphs, dims, focusedPath) {
  if (!rootNodes?.length) return { allNodes: [], positions: {}, edgePaths: [], totalW: 0, totalH: 0, subGraphBounds: {} };

  const subDims = { nodeH: dims.nodeH * SUB_SCALE, nodeW: dims.nodeW * SUB_SCALE, rowGap: dims.rowGap * SUB_SCALE };
  const rootLayout = computeLayout(rootNodes, rootEdges, dims);

  const allNodes = rootNodes.map(n => ({ ...n, _qualifiedId: n.id, _depth: 0 }));
  const positions = { ...rootLayout.positions };
  const edgePaths = [...rootLayout.edgePaths];
  const subGraphBounds = {};
  const subGroupedColumns = {};
  let maxW = rootLayout.totalW;
  let maxH = rootLayout.totalH;

  // Only include sub-graph nodes for the focused container (not all)
  const expandedParent = focusedPath?.length > 0 ? focusedPath[0] : null;

  if (subGraphs) {
    for (const [parentId, sg] of Object.entries(subGraphs)) {
      if (!sg?.nodes?.length) continue;
      if (parentId !== expandedParent) continue;
      const parentPos = positions[parentId];
      if (!parentPos) continue;

      const subLayout = computeLayout(sg.nodes, sg.edges, subDims);
      const offsetX = parentPos.x + dims.nodeW + SUB_GAP;
      const offsetY = parentPos.y - (subLayout.totalH - dims.nodeH) / 2;

      for (const sn of sg.nodes) {
        const sp = subLayout.positions[sn.id];
        if (!sp) continue;
        const qid = `${parentId}/${sn.id}`;
        positions[qid] = {
          x: offsetX + sp.x, y: offsetY + sp.y,
          cx: offsetX + sp.cx, cy: offsetY + sp.cy,
        };
        allNodes.push({ ...sn, id: qid, _qualifiedId: qid, _depth: 1, _parentId: parentId });
      }

      // Collect sub-graph grouped columns with offset
      for (const [colIdx, group] of Object.entries(subLayout.groupedColumns || {})) {
        const gcKey = `${parentId}/${colIdx}`;
        subGroupedColumns[gcKey] = {
          ...group,
          x: offsetX + group.x,
          y: offsetY + group.y,
          nodeIds: new Set([...group.nodeIds].map(id => `${parentId}/${id}`)),
          nodes: group.nodes.map(n => ({ ...n, id: `${parentId}/${n.id}`, _qualifiedId: `${parentId}/${n.id}`, _depth: 1, _parentId: parentId })),
        };
      }

      for (const ep of subLayout.edgePaths) {
        const [srcId, tgtId] = ep.key.split('-');
        const srcPos = positions[`${parentId}/${srcId}`];
        const tgtPos = positions[`${parentId}/${tgtId}`];
        if (srcPos && tgtPos) {
          const sx = srcPos.x + subDims.nodeW, sy = srcPos.cy;
          const tx = tgtPos.x, ty = tgtPos.cy;
          const cp = (tx - sx) * 0.5;
          edgePaths.push({
            key: `${parentId}/${srcId}-${parentId}/${tgtId}`,
            d: `M ${sx} ${sy} C ${sx + cp} ${sy}, ${tx - cp} ${ty}, ${tx} ${ty}`,
            _depth: 1, _parentId: parentId,
          });
        }
      }

      // Connector edge from container to sub-graph
      const firstSubNode = sg.nodes[0];
      if (firstSubNode) {
        const fqid = `${parentId}/${firstSubNode.id}`;
        const fpos = positions[fqid];
        if (fpos) {
          const sx = parentPos.x + dims.nodeW, sy = parentPos.cy;
          const tx = fpos.x, ty = fpos.cy;
          const cp = (tx - sx) * 0.5;
          edgePaths.push({
            key: `connector-${parentId}`,
            d: `M ${sx} ${sy} C ${sx + cp} ${sy}, ${tx - cp} ${ty}, ${tx} ${ty}`,
            _isConnector: true, _parentId: parentId,
          });
        }
      }

      subGraphBounds[parentId] = {
        x: offsetX, y: offsetY,
        w: subLayout.totalW, h: subLayout.totalH,
      };
      maxW = Math.max(maxW, offsetX + subLayout.totalW + PADDING);
      maxH = Math.max(maxH, offsetY + subLayout.totalH + PADDING);
    }
  }

  return { allNodes, positions, edgePaths, totalW: maxW, totalH: Math.max(maxH, PADDING * 2), subGraphBounds, groupedColumns: subGroupedColumns };
}

function StatusDot({ node, cx, cy, r, selected, onClick }) {
  const cfg = STATUS_CONFIG[node.status] || STATUS_CONFIG.pending;
  const isContainer = node.is_container;
  return (
    <g onClick={(e) => { e.stopPropagation(); onClick(node._qualifiedId || node.id); }} style={{ cursor: 'pointer' }}>
      <title>{node.label} — {node.status || 'pending'}</title>
      {isContainer ? (
        <rect x={cx - r} y={cy - r} width={r * 2} height={r * 2} rx={4}
          fill={cfg.color} opacity={selected ? 1 : 0.7}
          stroke={selected ? '#fff' : 'none'} strokeWidth={selected ? 2 : 0} />
      ) : (
        <circle cx={cx} cy={cy} r={r} fill={cfg.color} opacity={selected ? 1 : 0.7}
          stroke={selected ? '#fff' : 'none'} strokeWidth={selected ? 2 : 0} />
      )}
    </g>
  );
}

function MiniCard({ node, y, width, selected, onClick }) {
  const cfg = STATUS_CONFIG[node.status] || STATUS_CONFIG.pending;
  return (
    <g onClick={(e) => { e.stopPropagation(); onClick(node._qualifiedId || node.id); }} style={{ cursor: 'pointer' }}>
      <rect x={4} y={y} width={width} height={MINI_CARD_H - 4} rx={4}
        fill={selected ? 'rgba(25,118,210,0.2)' : 'rgba(255,255,255,0.05)'}
        stroke={selected ? '#1976d2' : cfg.color} strokeWidth={1} />
      {cfg.icon === 'spinner' && (
        <circle cx={16} cy={y + (MINI_CARD_H - 4) / 2} r={4}
          fill="none" stroke={cfg.color} strokeWidth={1.5} strokeDasharray="6 4">
          <animateTransform attributeName="transform" type="rotate"
            from={`0 16 ${y + (MINI_CARD_H - 4) / 2}`} to={`360 16 ${y + (MINI_CARD_H - 4) / 2}`}
            dur="1s" repeatCount="indefinite" />
        </circle>
      )}
      {cfg.icon === 'check' && (
        <text x={14} y={y + (MINI_CARD_H - 4) / 2 + 4} fontSize={10} fill={cfg.color}>✓</text>
      )}
      <text x={28} y={y + (MINI_CARD_H - 4) / 2 + 4} fontSize={10} fill="rgba(255,255,255,0.85)"
        clipPath={`inset(0 0 0 0)`}>
        <title>{node.label}</title>
        {node.label?.length > 18 ? node.label.slice(0, 17) + '…' : node.label}
      </text>
      {node.is_container && (
        <text x={width - 8} y={y + (MINI_CARD_H - 4) / 2 + 3} fontSize={8} fill="rgba(255,255,255,0.4)">⊞</text>
      )}
    </g>
  );
}

function GroupedColumn({ nodes, selectedId, onSelect, x, y, width, height }) {
  const running = nodes.filter(n => n.status === 'running');
  const pending = nodes.filter(n => n.status === 'pending');
  const completed = nodes.filter(n => n.status !== 'running' && n.status !== 'pending');
  [running, pending, completed].forEach(arr => arr.sort((a, b) => (a.id || '').localeCompare(b.id || '')));

  const dotStartY = GROUP_HEADER_H + running.length * MINI_CARD_H + 8;
  const allDots = [...completed, ...pending];

  return (
    <Box sx={{ position: 'absolute', left: x, top: y, width, height }}>
      <svg width={width} height={height} style={{ overflow: 'visible' }}>
        <rect width={width} height={height} rx={8} fill="rgba(255,255,255,0.03)"
          stroke="rgba(255,255,255,0.12)" strokeWidth={1} />
        <text x={8} y={16} fontSize={10} fill="rgba(255,255,255,0.5)" fontFamily="sans-serif">
          {nodes.length} nodes
          {'  '}✅{completed.length} ⏳{running.length} ⏸{pending.length}
        </text>
        {running.map((node, i) => (
          <MiniCard key={node.id} node={node} y={GROUP_HEADER_H + i * MINI_CARD_H}
            width={width - 8} selected={selectedId === (node._qualifiedId || node.id)} onClick={onSelect} />
        ))}
        {allDots.map((node, i) => (
          <StatusDot key={node.id} node={node}
            cx={DOT_SIZE / 2 + 4 + (i % DOTS_PER_ROW) * DOT_STRIDE}
            cy={dotStartY + Math.floor(i / DOTS_PER_ROW) * DOT_STRIDE}
            r={DOT_SIZE / 2 - 2}
            selected={selectedId === (node._qualifiedId || node.id)} onClick={onSelect} />
        ))}
      </svg>
    </Box>
  );
}

function NodeBox({ node, position, isSelected, isExpandable, compact, dims, opacity, badge, onClick }) {
  const theme = useTheme();
  const cfg = STATUS_CONFIG[node.status] || STATUS_CONFIG.pending;
  const borderColor = isSelected ? theme.palette.primary.main : cfg.color;
  const bgColor = isSelected
    ? `${theme.palette.primary.main}22`
    : node.status === 'running' ? `${cfg.color}11` : 'transparent';

  const nodeW = dims?.nodeW || NORMAL.nodeW;
  const nodeH = dims?.nodeH || NORMAL.nodeH;
  const isSmall = nodeH < 50;

  return (
    <Box
      onClick={(e) => { e.stopPropagation(); onClick(node._qualifiedId || node.id); }}
      sx={{
        position: 'absolute',
        left: position.x, top: position.y,
        width: nodeW, height: nodeH,
        borderRadius: isSmall ? 1 : 1.5,
        border: `${isSmall ? 1 : 2}px solid ${borderColor}`,
        backgroundColor: bgColor,
        cursor: 'pointer',
        display: 'flex',
        flexDirection: isSmall ? 'row' : 'column',
        alignItems: isSmall ? 'center' : 'flex-start',
        justifyContent: isSmall ? 'flex-start' : 'center',
        px: isSmall ? 0.5 : 1.25,
        py: isSmall ? 0 : 0.75,
        gap: isSmall ? 0.3 : 0.5,
        opacity,
        transition: 'opacity 0.3s ease, border-color 0.2s, background-color 0.2s',
        '&:hover': { borderColor: theme.palette.primary.main, opacity: Math.max(opacity, 0.8) },
        boxSizing: 'border-box',
        overflow: 'hidden',
      }}
    >
      {!isSmall && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, width: '100%' }}>
          {cfg.icon === 'spinner' && <CircularProgress size={12} sx={{ color: cfg.color, flexShrink: 0 }} />}
          {cfg.icon === 'check' && <CheckCircleIcon sx={{ fontSize: 13, color: cfg.color, flexShrink: 0 }} />}
          {cfg.icon === 'error' && <ErrorIcon sx={{ fontSize: 13, color: 'error.main', flexShrink: 0 }} />}
          <Typography variant="caption" sx={{ fontSize: '0.7rem', color: cfg.color, fontWeight: 600, textTransform: 'capitalize', flex: 1 }}>
            {node.status || 'pending'}
          </Typography>
          {isExpandable && <AccountTreeIcon sx={{ fontSize: 14, color: 'text.disabled', flexShrink: 0 }} />}
        </Box>
      )}
      {isSmall && cfg.icon === 'spinner' && <CircularProgress size={8} sx={{ color: cfg.color, flexShrink: 0 }} />}
      {isSmall && cfg.icon === 'check' && <CheckCircleIcon sx={{ fontSize: 9, color: cfg.color, flexShrink: 0 }} />}
      {isSmall && cfg.icon === 'error' && <ErrorIcon sx={{ fontSize: 9, color: 'error.main', flexShrink: 0 }} />}
      {isSmall && isExpandable && <AccountTreeIcon sx={{ fontSize: 9, color: 'text.disabled', flexShrink: 0 }} />}

      <Typography variant="caption" sx={{
        fontSize: isSmall ? '0.58rem' : '0.72rem', color: 'text.primary',
        fontWeight: isSelected ? 600 : 400,
        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        width: '100%', lineHeight: 1.3,
      }} title={node.label}>
        {node.label}
      </Typography>

      {badge && !isSmall && (
        <Typography variant="caption" sx={{ fontSize: '0.6rem', color: 'text.disabled', mt: -0.25 }}>
          {badge}
        </Typography>
      )}
    </Box>
  );
}

function GhostOutlines({ ghostNodes, containerRect, onGhostClick }) {
  const theme = useTheme();
  if (!ghostNodes?.length || !containerRect) return null;

  return (
    <>
      {ghostNodes.map(g => (
        <Tooltip key={g.id} title={`${g.label} — ${g.status || 'pending'}`} arrow placement="top">
          <Box
            onClick={() => onGhostClick(g.id)}
            sx={{
              position: 'absolute',
              ...(g.edge === 'left' ? { left: 4, top: g.posPercent + '%' } :
                  g.edge === 'right' ? { right: 4, top: g.posPercent + '%' } :
                  g.edge === 'top' ? { top: 4, left: g.posPercent + '%' } :
                  { bottom: 4, left: g.posPercent + '%' }),
              width: GHOST_SIZE, height: GHOST_SIZE,
              borderRadius: 1,
              border: `1px solid ${(STATUS_CONFIG[g.status] || STATUS_CONFIG.pending).color}55`,
              backgroundColor: `${(STATUS_CONFIG[g.status] || STATUS_CONFIG.pending).color}15`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              cursor: 'pointer', zIndex: 5,
              '&:hover': { backgroundColor: `${theme.palette.primary.main}30`, borderColor: theme.palette.primary.main },
            }}
          >
            <Typography variant="caption" sx={{ fontSize: '0.5rem', color: 'text.secondary', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 28, textAlign: 'center' }}>
              {(g.label || g.id).slice(0, 4)}
            </Typography>
          </Box>
        </Tooltip>
      ))}
    </>
  );
}

export function GraphFlowView({
  nodes, edges, selectedNodeId, expandableNodeIds, onNodeClick,
  viewMode = 'page-switch', subGraphs, focusedPath = [], onFocusChange,
}) {
  const theme = useTheme();
  const containerRef = useRef(null);

  const maxColSize = useMemo(() => {
    if (!nodes?.length) return 0;
    const inDegree = {}, outEdges = {};
    nodes.forEach(n => { inDegree[n.id] = 0; outEdges[n.id] = []; });
    (edges || []).forEach(e => { if (outEdges[e.source]) outEdges[e.source].push(e.target); if (e.target in inDegree) inDegree[e.target]++; });
    const depth = {};
    const queue = nodes.filter(n => !inDegree[n.id]).map(n => n.id);
    queue.forEach(id => { depth[id] = 0; });
    const visited = new Set(queue);
    let head = 0;
    while (head < queue.length) { const cur = queue[head++]; (outEdges[cur] || []).forEach(child => { const d = (depth[cur] || 0) + 1; if (depth[child] === undefined || depth[child] < d) depth[child] = d; if (!visited.has(child)) { visited.add(child); queue.push(child); } }); }
    nodes.forEach(n => { if (depth[n.id] === undefined) depth[n.id] = 0; });
    const counts = {};
    nodes.forEach(n => { const d = depth[n.id]; counts[d] = (counts[d] || 0) + 1; });
    return Math.max(...Object.values(counts));
  }, [nodes, edges]);

  const compact = maxColSize > 8;
  const dims = compact ? COMPACT : NORMAL;

  const isFocusContext = viewMode === 'focus-context' && subGraphs;

  const { allNodes: unifiedNodes, positions: unifiedPositions, edgePaths: unifiedEdgePaths, totalW: unifiedTotalW, totalH: unifiedTotalH, subGraphBounds, groupedColumns: unifiedGroupedColumns } = useMemo(() => {
    if (!isFocusContext) return { allNodes: [], positions: {}, edgePaths: [], totalW: 0, totalH: 0, subGraphBounds: {}, groupedColumns: {} };
    return computeUnifiedLayout(nodes, edges, subGraphs, dims, focusedPath);
  }, [isFocusContext, nodes, edges, subGraphs, dims, focusedPath]);

  const { positions: pageSwitchPositions, totalW: psTotalW, totalH: psTotalH, edgePaths: psEdgePaths, groupedColumns: psGroupedColumns } = useMemo(() => {
    if (isFocusContext) return { positions: {}, totalW: 0, totalH: 0, edgePaths: [], groupedColumns: {} };
    return computeLayout(nodes, edges, dims);
  }, [isFocusContext, nodes, edges, dims]);

  const activeNodes = isFocusContext ? unifiedNodes : nodes;
  const activePositions = isFocusContext ? unifiedPositions : pageSwitchPositions;
  const activeEdgePaths = isFocusContext ? unifiedEdgePaths : psEdgePaths;
  const totalW = isFocusContext ? unifiedTotalW : psTotalW;
  const totalH = isFocusContext ? unifiedTotalH : psTotalH;

  // Zoom/pan
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [animating, setAnimating] = useState(false);
  const dragRef = useRef({ dragging: false, startX: 0, startY: 0, startPanX: 0, startPanY: 0 });

  const fitToView = useCallback((bounds) => {
    if (!containerRef.current || !totalW || !totalH) return;
    const rect = containerRef.current.getBoundingClientRect();
    const target = bounds || { x: 0, y: 0, w: totalW, h: totalH };
    const scaleX = rect.width / target.w;
    const scaleY = rect.height / target.h;
    const scale = Math.min(scaleX, scaleY, 1.5) * 0.88;
    const centerX = target.x + target.w / 2;
    const centerY = target.y + target.h / 2;
    setAnimating(true);
    setZoom(Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, scale)));
    setPan({ x: rect.width / 2 - centerX * scale, y: rect.height / 2 - centerY * scale });
    setTimeout(() => setAnimating(false), 650);
  }, [totalW, totalH]);

  // Auto-fit on topology change or focus change
  const nodeCount = activeNodes?.length || 0;
  const focusKey = focusedPath.join('/');
  useEffect(() => {
    if (isFocusContext && focusKey && subGraphBounds[focusKey]) {
      fitToView(subGraphBounds[focusKey]);
    } else {
      fitToView();
    }
  }, [nodeCount, focusKey, isFocusContext, fitToView, subGraphBounds]);

  const handleWheel = useCallback((e) => {
    e.preventDefault();
    setZoom(z => Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z + (e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP))));
  }, []);

  const handleMouseDown = useCallback((e) => {
    if (e.button !== 0 || e.target.closest('[data-node]')) return;
    dragRef.current = { dragging: true, startX: e.clientX, startY: e.clientY, startPanX: pan.x, startPanY: pan.y };
  }, [pan]);

  const handleMouseMove = useCallback((e) => {
    if (!dragRef.current.dragging) return;
    setPan({ x: dragRef.current.startPanX + (e.clientX - dragRef.current.startX), y: dragRef.current.startPanY + (e.clientY - dragRef.current.startY) });
  }, []);

  const handleMouseUp = useCallback(() => { dragRef.current.dragging = false; }, []);

  // Focus-context: compute node opacity and badges
  const getNodeOpacity = useCallback((nodeId) => {
    if (!isFocusContext || focusedPath.length === 0) return 1;
    const prefix = focusedPath.join('/');
    if (nodeId === prefix || nodeId.startsWith(prefix + '/')) return 1;
    // Root nodes that are siblings of the focused container
    if (!nodeId.includes('/')) return 0.15;
    return 0.15;
  }, [isFocusContext, focusedPath]);

  const containerBadges = useMemo(() => {
    if (!isFocusContext || !subGraphs) return {};
    const badges = {};
    for (const [parentId, sg] of Object.entries(subGraphs)) {
      if (!sg?.nodes) continue;
      const done = sg.nodes.filter(n => n.status === 'completed' || n.status === 'skipped').length;
      const running = sg.nodes.filter(n => n.status === 'running').length;
      const total = sg.nodes.length;
      badges[parentId] = `▶ ${done}/${total} ✓${running ? ` · ${running} ⚙` : ''}`;
    }
    return badges;
  }, [isFocusContext, subGraphs]);

  // Ghost outlines for non-focused root nodes
  const ghostNodes = useMemo(() => {
    if (!isFocusContext || focusedPath.length === 0 || !nodes?.length) return [];
    const focused = focusedPath[0];
    return nodes.filter(n => n.id !== focused).map(n => {
      const pos = activePositions[n.id];
      if (!pos) return null;
      const focusedBounds = subGraphBounds[focusedPath.join('/')];
      if (!focusedBounds) return null;
      const relX = pos.x < focusedBounds.x ? 'left' : 'right';
      const posPercent = Math.max(5, Math.min(90, ((pos.y / totalH) * 100)));
      return { id: n.id, label: n.label, status: n.status, edge: relX, posPercent };
    }).filter(Boolean);
  }, [isFocusContext, focusedPath, nodes, activePositions, subGraphBounds, totalH]);

  const handleGhostClick = useCallback((nodeId) => {
    if (onFocusChange) onFocusChange([]);
  }, [onFocusChange]);

  const expandable = expandableNodeIds instanceof Set ? expandableNodeIds : new Set(expandableNodeIds || []);
  const activeGroupedColumns = isFocusContext
    ? (unifiedGroupedColumns || {})
    : (psGroupedColumns || {});
  const groupedNodeIds = useMemo(() => {
    const ids = new Set();
    for (const g of Object.values(activeGroupedColumns)) {
      if (g.nodeIds) g.nodeIds.forEach(id => ids.add(id));
    }
    return ids;
  }, [activeGroupedColumns]);

  // Determine which node dims to use for each node
  const getNodeDims = useCallback((node) => {
    if (isFocusContext && node._depth > 0) {
      return { nodeH: dims.nodeH * SUB_SCALE, nodeW: dims.nodeW * SUB_SCALE, rowGap: dims.rowGap * SUB_SCALE };
    }
    return dims;
  }, [isFocusContext, dims]);

  if (!nodes || !nodes.length) {
    return <Box sx={{ p: 2, color: 'text.secondary' }}><Typography variant="caption">No graph data.</Typography></Box>;
  }

  return (
    <Box
      ref={containerRef}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onClick={() => { if (isFocusContext && focusedPath.length > 0) onFocusChange?.([]); }}
      sx={{
        position: 'relative',
        width: '100%',
        height: Math.max(totalH || 200, 200),
        overflow: 'hidden',
        cursor: dragRef.current?.dragging ? 'grabbing' : 'grab',
        userSelect: 'none',
      }}
    >
      {/* Zoom controls */}
      <Box sx={{ position: 'absolute', top: 4, right: 4, zIndex: 10, display: 'flex', gap: 0.25, backgroundColor: 'rgba(0,0,0,0.4)', borderRadius: 1, px: 0.25 }}>
        <IconButton size="small" onClick={() => setZoom(z => Math.min(MAX_ZOOM, z + ZOOM_STEP))} sx={{ color: 'text.secondary', p: 0.5 }} title="Zoom in"><ZoomInIcon sx={{ fontSize: 16 }} /></IconButton>
        <IconButton size="small" onClick={() => setZoom(z => Math.max(MIN_ZOOM, z - ZOOM_STEP))} sx={{ color: 'text.secondary', p: 0.5 }} title="Zoom out"><ZoomOutIcon sx={{ fontSize: 16 }} /></IconButton>
        <IconButton size="small" onClick={() => fitToView()} sx={{ color: 'text.secondary', p: 0.5 }} title="Fit to view"><FitScreenIcon sx={{ fontSize: 16 }} /></IconButton>
      </Box>

      {/* Ghost outlines */}
      {isFocusContext && <GhostOutlines ghostNodes={ghostNodes} containerRect={containerRef.current?.getBoundingClientRect()} onGhostClick={handleGhostClick} />}

      {/* Transformed canvas */}
      <Box sx={{
        position: 'absolute', width: totalW, height: totalH,
        transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
        transformOrigin: '0 0',
        transition: animating ? 'transform 0.6s cubic-bezier(0.4, 0, 0.2, 1)' : (dragRef.current?.dragging ? 'none' : 'transform 0.15s ease-out'),
      }}>
        <svg style={{ position: 'absolute', top: 0, left: 0, width: totalW, height: totalH, pointerEvents: 'none' }}>
          {(() => {
            // Build nodeId → group lookup for edge rerouting
            const nodeToGroup = {};
            for (const g of Object.values(activeGroupedColumns)) {
              if (g.nodeIds) g.nodeIds.forEach(id => { nodeToGroup[id] = g; });
            }
            // Deduplicate edges between same group pairs
            const seen = new Set();
            return activeEdgePaths.map(ep => {
              const [srcId, tgtId] = ep.key.split('-');
              const srcGrp = nodeToGroup[srcId];
              const tgtGrp = nodeToGroup[tgtId];
              if (srcGrp && tgtGrp && srcGrp === tgtGrp) return null; // skip intra-group
              const dedupKey = `${srcGrp ? 'g' + srcGrp.x : srcId}-${tgtGrp ? 'g' + tgtGrp.x : tgtId}`;
              if (srcGrp || tgtGrp) {
                if (seen.has(dedupKey)) return null;
                seen.add(dedupKey);
              }
              // Reroute endpoints to group box center
              let d = ep.d;
              if (srcGrp || tgtGrp) {
                const src = activePositions[srcId];
                const tgt = activePositions[tgtId];
                if (src && tgt) {
                  const sx = srcGrp ? srcGrp.x + srcGrp.w : src.x + (dims?.nodeW || NORMAL.nodeW);
                  const sy = srcGrp ? srcGrp.y + srcGrp.h / 2 : src.cy;
                  const tx = tgtGrp ? tgtGrp.x : tgt.x;
                  const ty = tgtGrp ? tgtGrp.y + tgtGrp.h / 2 : tgt.cy;
                  const cp = (tx - sx) * 0.5;
                  d = `M ${sx} ${sy} C ${sx + cp} ${sy}, ${tx - cp} ${ty}, ${tx} ${ty}`;
                }
              }
              const edgeOpacity = isFocusContext && focusedPath.length > 0
                ? (ep._parentId && focusedPath.join('/') === ep._parentId ? 1 : (ep._isConnector && focusedPath[0] === ep._parentId ? 0.4 : 0.1))
                : 1;
              return (
                <path key={ep.key} d={d} fill="none" stroke={theme.palette.divider}
                  strokeWidth={(srcGrp || tgtGrp) ? 2 : (ep._depth > 0 || compact ? 1 : 1.5)}
                  style={{ opacity: edgeOpacity, transition: 'opacity 0.3s ease' }}
                  strokeDasharray={ep._isConnector ? '4 3' : undefined} />
              );
            }).filter(Boolean);
          })()}
        </svg>

        {/* Grouped columns */}
        {Object.entries(activeGroupedColumns).map(([colIdx, group]) => (
          <Box key={`group-${colIdx}`} data-node>
            <GroupedColumn
              nodes={group.nodes}
              selectedId={selectedNodeId}
              onSelect={onNodeClick}
              x={group.x} y={group.y}
              width={group.w} height={group.h}
            />
          </Box>
        ))}

        {/* Individual nodes (skip those in grouped columns) */}
        {(isFocusContext ? unifiedNodes : nodes).map(node => {
          const nodeId = node._qualifiedId || node.id;
          if (groupedNodeIds.has(nodeId)) return null;
          const pos = activePositions[nodeId];
          if (!pos) return null;
          const nodeDims = getNodeDims(node);
          const isExp = isFocusContext ? !!subGraphs?.[nodeId] : expandable.has(nodeId);
          const opacity = getNodeOpacity(nodeId);
          return (
            <Box key={nodeId} data-node>
              <NodeBox
                node={node}
                position={pos}
                isSelected={selectedNodeId === nodeId}
                isExpandable={isExp}
                compact={false}
                dims={nodeDims}
                opacity={opacity}
                badge={containerBadges[nodeId]}
                onClick={onNodeClick}
              />
            </Box>
          );
        })}
      </Box>
    </Box>
  );
}

export default GraphFlowView;

/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * ViewTabBar — Secondary tab strip for multi-view task panels.
 * Shows view tabs with progressive unlock (disabled + dimmed for locked)
 * and notification dot for newly-unlocked views.
 */

import React from 'react';
import { Box, Tab, Tabs, Badge } from '@mui/material';

export function ViewTabBar({ views, activeIndex, onSwitch }) {
  return (
    <Box sx={{ borderBottom: '1px solid', borderColor: 'divider' }}>
      <Tabs
        value={activeIndex}
        onChange={(_, newIndex) => {
          const view = views[newIndex];
          if (view?.isUnlocked) onSwitch(newIndex);
        }}
        variant="scrollable"
        scrollButtons="auto"
        sx={{
          minHeight: 36,
          '& .MuiTab-root': {
            textTransform: 'none',
            fontSize: '0.82rem',
            minHeight: 36,
            py: 0.5,
          },
        }}
      >
        {views.map((view, i) => (
          <Tab
            key={i}
            label={
              <Badge
                variant="dot"
                color="primary"
                invisible={!view.isUnlocked || i === activeIndex}
                sx={{ '& .MuiBadge-dot': { top: 2, right: -4 } }}
              >
                {view.label}
              </Badge>
            }
            disabled={!view.isUnlocked}
            sx={{
              opacity: view.isUnlocked ? 1 : 0.4,
              transition: 'opacity 0.3s',
            }}
          />
        ))}
      </Tabs>
    </Box>
  );
}

export default ViewTabBar;

/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * WidgetHostView — generic dashboard view that hosts one or more widgets from
 * the shared WidgetRegistry. This is the seam behind "any conversation UI tool
 * can be a tab in a dashboard": a view spec names `widgets: [{type, configSource,
 * ...}]`, and each widget's config is built from `dashboardState.scenarioState`.
 *
 * Generic + injectable: the hosting dashboard may pass
 *   - `enrichConfig(builtConfig, spec, state)` → layer domain-specific config
 *     (e.g. the Experiment Hub's incremental-submit locking), and
 *   - `onWidgetSubmit(spec, response)` → route a widget submit (e.g. to the
 *     hub's `runImplementHypothesis`), and
 *   - `renderHeader(state, config)` → an optional header (e.g. batch summary).
 */

import React, { useMemo } from 'react';
import { Box, Typography } from '@mui/material';
import { getWidget } from '../protocol/WidgetRegistry';
import { deriveStatusMap } from './lifecycle';

const noop = () => {};

function buildWidgetConfig(dashboardState, spec, enrichConfig) {
  const scenarioState = (dashboardState && dashboardState.scenarioState) || {};
  const source = (spec.configSource && scenarioState[spec.configSource]) || spec.config || {};
  let config = {
    input_mode: {
      metadata: source.widgetConfig || source.metadata || source,
      prompt: source.prompt || '',
      options: source.options,
    },
    _selectedProposals: source.selectedValues || source.selectedProposals || [],
    _customQueries: source.customQueries || [],
    _dashboardId: (dashboardState && dashboardState.id) || null,
    _submitted: spec.readOnly || false,
  };
  if (spec.enrichments && spec.enrichments.statusMapSource) {
    config._implementationStatus = deriveStatusMap(
      dashboardState && dashboardState[spec.enrichments.statusMapSource]
    );
  }
  if (typeof enrichConfig === 'function') {
    config = enrichConfig(config, spec, dashboardState) || config;
  }
  return config;
}

export default function WidgetHostView({
  dashboardState,
  config = {},
  onWidgetSubmit,
  enrichConfig,
  renderHeader,
}) {
  const widgets = useMemo(
    () => (config.widgets || []).map((spec) => ({
      spec,
      builtConfig: buildWidgetConfig(dashboardState, spec, enrichConfig),
      Widget: getWidget(spec.type),
    })),
    [dashboardState, config.widgets, enrichConfig]
  );

  return (
    <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
      {typeof renderHeader === 'function' && renderHeader(dashboardState, config)}
      {widgets.map(({ spec, builtConfig, Widget }, i) => (
        <Box key={i} sx={{ mb: 2 }}>
          {Widget ? (
            <Widget
              config={builtConfig}
              onSubmit={(response) => (onWidgetSubmit ? onWidgetSubmit(spec, response) : noop())}
            />
          ) : (
            <Typography color="text.secondary">Unknown widget type: {spec.type}</Typography>
          )}
        </Box>
      ))}
    </Box>
  );
}

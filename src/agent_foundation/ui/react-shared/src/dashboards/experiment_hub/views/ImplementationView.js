/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * ImplementationView — the Experiment Hub's "Implementation" tab. Composes the
 * ported QueueProgressView (← RunQueueList + RunStreamPanel). Maps the generic
 * dashboard view props (`dashboardState`) to the ported component's `task`
 * prop and threads the injected `apiClient` (for the Resume affordance) +
 * `dispatch` (for SELECT_QUEUE_ENTRY).
 */

import React from 'react';
import QueueProgressView from '../components/views/QueueProgressView';

export default function ImplementationView({
  dashboardState, config, dispatch, apiClient,
}) {
  return (
    <QueueProgressView
      task={dashboardState}
      config={config || {}}
      dispatch={dispatch}
      apiClient={apiClient}
    />
  );
}

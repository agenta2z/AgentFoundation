/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * ReviewComboView — the Experiment Hub's "Review & Combo" tab. Composes the
 * ported MultiChoiceComboView (← ComboReviewSection + HypothesisDetailDrawer +
 * SubmissionFooterBar + setup modals/drawers). Maps the generic dashboard view
 * props to the ported component's `task` prop and threads `sessionId`,
 * `dispatch`, and the injected `apiClient`.
 */

import React from 'react';
import MultiChoiceComboView from '../components/views/MultiChoiceComboView';

export default function ReviewComboView({
  dashboardState, config, dispatch, sessionId, apiClient,
}) {
  return (
    <MultiChoiceComboView
      task={dashboardState}
      config={config || {}}
      dispatch={dispatch}
      sessionId={sessionId}
      apiClient={apiClient}
    />
  );
}

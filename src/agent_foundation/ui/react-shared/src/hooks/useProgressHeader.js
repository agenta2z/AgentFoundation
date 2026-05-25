/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Custom hook for managing progress header state.
 * Handles input during progress_header phase and submission.
 */

import { useState, useCallback } from 'react';
import { API_BASE } from '../utils/api';
import { addDebugLog } from '../utils/debug';

/**
 * Hook for managing progress header state
 * @param {Function} startProgressPolling - Callback to start polling after continue
 * @returns {object} Progress header state and handlers
 */
export function useProgressHeader(startProgressPolling) {
  // Progress header input value
  const [progressHeaderInput, setProgressHeaderInput] = useState('');
  // Track if progress header has been submitted
  const [progressHeaderSubmitted, setProgressHeaderSubmitted] = useState(false);
  // Store submitted data to continue showing it in disabled state
  const [submittedProgressHeader, setSubmittedProgressHeader] = useState(null);

  const handleProgressHeaderContinue = useCallback(async (progressState, callbacks) => {
    const { setProgressState, setAnimationPhase, setIsAnimating, pollingIntervalRef } = callbacks;

    addDebugLog('progress_header', 'handleProgressHeaderContinue called', {
      phase: progressState?.phase,
      hasProgressHeader: !!progressState?.progress_header,
      progressHeaderInput: progressHeaderInput?.substring(0, 50),
      pollingActive: !!pollingIntervalRef?.current,
    });

    if (!progressState || progressState.phase !== 'progress_header') {
      addDebugLog('progress_header', 'Early return - phase not progress_header', {
        phase: progressState?.phase,
      });
      return;
    }

    // Save submitted data BEFORE API call to show in disabled state
    setSubmittedProgressHeader({
      header: progressState.progress_header,
      inputValue: progressHeaderInput,
    });
    setProgressHeaderSubmitted(true);

    try {
      // Build user_input from progressHeaderInput if we have an input field
      const userInput = {};
      if (progressState.progress_header?.input_field) {
        const variableName = progressState.progress_header.input_field.variable_name;
        userInput[variableName] = progressHeaderInput;
      }

      addDebugLog('progress_header', 'Calling /api/chat/progress/continue', { userInput });

      const response = await fetch(`${API_BASE}/chat/progress/continue`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_input: userInput }),
      });

      if (!response.ok) {
        addDebugLog('progress_header', 'API error', { status: response.status });
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const progressData = await response.json();

      addDebugLog('progress_header', 'API response received', {
        phase: progressData.phase,
        is_animating: progressData.is_animating,
        sections_count: progressData.sections?.length || 0,
        waiting_for_progress_header: progressData.waiting_for_progress_header,
      });

      // Update state
      setProgressState(progressData);
      setAnimationPhase(progressData.phase);

      // Start polling if animation is continuing
      if (progressData.is_animating) {
        addDebugLog('progress_header', 'Animation active, starting polling directly', {
          currentPollingActive: !!pollingIntervalRef?.current,
        });
        setIsAnimating(true);

        // Clear any existing polling first
        if (pollingIntervalRef?.current) {
          addDebugLog('progress_header', 'Clearing existing polling interval', {});
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
        }

        // Start polling immediately
        addDebugLog('progress_header', 'Calling startProgressPolling() now', {});
        startProgressPolling();
      } else {
        addDebugLog('progress_header', 'Animation not active, not starting polling', {});
      }
    } catch (error) {
      addDebugLog('progress_header', 'Error continuing from progress_header', { error: error.message });
      console.error('Error continuing from progress_header:', error);
      // Reset submitted state on error so user can try again
      setProgressHeaderSubmitted(false);
      setSubmittedProgressHeader(null);
    }
  }, [progressHeaderInput, startProgressPolling]);

  const resetProgressHeader = useCallback(() => {
    setProgressHeaderInput('');
    setProgressHeaderSubmitted(false);
    setSubmittedProgressHeader(null);
  }, []);

  const initializeProgressHeaderInput = useCallback((defaultValue) => {
    setProgressHeaderInput(current => {
      if (current === '' && defaultValue) {
        return defaultValue;
      }
      return current;
    });
  }, []);

  return {
    // State
    progressHeaderInput,
    progressHeaderSubmitted,
    submittedProgressHeader,
    // Actions
    setProgressHeaderInput,
    handleProgressHeaderContinue,
    resetProgressHeader,
    initializeProgressHeaderInput,
  };
}

export default useProgressHeader;

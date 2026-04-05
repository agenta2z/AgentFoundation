/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Custom hook for managing section visibility based on appearance delays.
 * Handles staggered section appearance during progress animations.
 */

import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * Hook for managing section visibility based on appearance_delay_ms
 * @param {object} progressState - Current progress state from backend
 * @param {string} animationPhase - Current animation phase
 * @param {string} lastProgressPhase - Previous animation phase (for detecting transitions)
 * @returns {object} Section visibility state and handlers
 */
export function useSectionVisibility(progressState, animationPhase, lastProgressPhase) {
  // Track which sections are visible based on their appearance_delay_ms
  const [visibleSections, setVisibleSections] = useState({});
  // Track when sections first became visible (for animation triggers)
  const [sectionAppearTimes, setSectionAppearTimes] = useState({});
  // Track when we first received progressState (for calculating elapsed time client-side)
  const [progressStartTimeClient, setProgressStartTimeClient] = useState(null);
  // Track whether we've already initialized for this progress phase (prevents re-initialization)
  const progressInitializedRef = useRef(false);

  // Update visible sections based on elapsed time and appearance_delay_ms
  useEffect(() => {
    if (!progressState || !progressState.sections || !progressState.is_animating) {
      // Reset the initialization flag when animation stops
      progressInitializedRef.current = false;
      return;
    }

    const currentPhase = progressState.phase;

    // Only initialize timer ONCE when entering progress phase
    if (currentPhase === 'progress' && !progressInitializedRef.current) {
      progressInitializedRef.current = true;
      setProgressStartTimeClient(Date.now());
      console.log('[useSectionVisibility] Initialized progress start time');
      return; // Wait for next render with the start time set
    }

    // Reset flag when leaving progress phase
    if (currentPhase !== 'progress' && currentPhase !== 'post_delay' &&
        currentPhase !== 'post_messages' && currentPhase !== 'complete') {
      progressInitializedRef.current = false;
      return;
    }

    if (progressStartTimeClient === null) {
      setProgressStartTimeClient(Date.now());
      return; // Wait for next render with the start time set
    }

    const updateVisibility = () => {
      const elapsed = Date.now() - progressStartTimeClient;
      const newVisible = {};
      const newAppearTimes = { ...sectionAppearTimes };
      let changed = false;

      progressState.sections.forEach(section => {
        const appearanceDelay = section.appearance_delay_ms || 0;
        const isVisible = elapsed >= appearanceDelay;
        newVisible[section.slot] = isVisible;

        // Track when section first became visible (for animation trigger)
        if (isVisible && !sectionAppearTimes[section.slot]) {
          newAppearTimes[section.slot] = Date.now();
          changed = true;
        }
      });

      setVisibleSections(newVisible);
      if (changed) {
        setSectionAppearTimes(newAppearTimes);
      }
    };

    // Initial check
    updateVisibility();

    // Set up interval to check visibility every 100ms
    const intervalId = setInterval(updateVisibility, 100);

    return () => clearInterval(intervalId);
  }, [progressStartTimeClient, progressState, lastProgressPhase, sectionAppearTimes]);

  // Reset visibility state when animation completes or new animation starts
  useEffect(() => {
    if (!progressState || !progressState.is_animating) {
      if (Object.keys(visibleSections).length > 0) {
        setVisibleSections({});
        setSectionAppearTimes({});
        setProgressStartTimeClient(null);
      }
    }
  }, [progressState?.is_animating, visibleSections]);

  const resetVisibility = useCallback(() => {
    setVisibleSections({});
    setSectionAppearTimes({});
    setProgressStartTimeClient(null);
  }, []);

  const isSectionVisible = useCallback((sectionSlot) => {
    return visibleSections[sectionSlot] === true;
  }, [visibleSections]);

  const getSectionAppearTime = useCallback((sectionSlot) => {
    return sectionAppearTimes[sectionSlot];
  }, [sectionAppearTimes]);

  const hasJustAppeared = useCallback((sectionSlot, thresholdMs = 1000) => {
    const appearTime = sectionAppearTimes[sectionSlot];
    return appearTime && (Date.now() - appearTime) < thresholdMs;
  }, [sectionAppearTimes]);

  return {
    // State
    visibleSections,
    sectionAppearTimes,
    progressStartTimeClient,
    // Actions
    resetVisibility,
    isSectionVisible,
    getSectionAppearTime,
    hasJustAppeared,
  };
}

export default useSectionVisibility;

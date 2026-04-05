/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Custom hook for managing progress animation state.
 * Handles progress phases, sections, and animation lifecycle.
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { addDebugLog } from '../utils/debug';

/**
 * Hook for managing progress animation state
 * @returns {object} Progress animation state and handlers
 */
export function useProgress() {
  // Progress state from backend
  const [progressState, setProgressState] = useState(null);
  // Whether animation is currently running
  const [isAnimating, setIsAnimating] = useState(false);
  // Collapsed sections tracking
  const [collapsedSections, setCollapsedSections] = useState({});
  // Track phase for detecting transitions (using ref to track previous value)
  const [lastProgressPhase, setLastProgressPhase] = useState(null);
  // Current animation phase
  const [animationPhase, setAnimationPhase] = useState('idle');

  // Phase-based animation state
  const [currentPreMessage, setCurrentPreMessage] = useState(null);
  const [revealedPostMessages, setRevealedPostMessages] = useState([]);

  // Use ref to track previous animationPhase value
  const prevAnimationPhaseRef = useRef(animationPhase);

  // Automatically update lastProgressPhase when animationPhase changes
  useEffect(() => {
    if (prevAnimationPhaseRef.current !== animationPhase) {
      addDebugLog('phase', 'Phase transition detected', {
        from: prevAnimationPhaseRef.current,
        to: animationPhase,
      });
      setLastProgressPhase(prevAnimationPhaseRef.current);
      prevAnimationPhaseRef.current = animationPhase;
    }
  }, [animationPhase]);

  // Debug: Log whenever progressState changes
  useEffect(() => {
    if (progressState) {
      addDebugLog('state', 'progressState changed', {
        phase: progressState.phase,
        is_animating: progressState.is_animating,
        sections_count: progressState.sections?.length || 0,
        keep_progress_sections: progressState.keep_progress_sections,
      });
    } else {
      addDebugLog('state', 'progressState changed', { value: 'null' });
    }
  }, [progressState]);

  const toggleSection = useCallback((sectionSlot) => {
    setCollapsedSections(prev => ({
      ...prev,
      [sectionSlot]: !prev[sectionSlot],
    }));
  }, []);

  const collapseSection = useCallback((sectionSlot) => {
    setCollapsedSections(prev => ({
      ...prev,
      [sectionSlot]: true,
    }));
  }, []);

  const collapseSections = useCallback((sectionSlots) => {
    setCollapsedSections(prev => {
      const newCollapsed = { ...prev };
      sectionSlots.forEach(slot => {
        newCollapsed[slot] = true;
      });
      return newCollapsed;
    });
  }, []);

  const resetAnimationState = useCallback(() => {
    setProgressState(null);
    setCurrentPreMessage(null);
    setRevealedPostMessages([]);
    setAnimationPhase('idle');
    setLastProgressPhase(null);
  }, []);

  const updatePhase = useCallback((phase) => {
    if (phase !== lastProgressPhase) {
      setLastProgressPhase(phase);
    }
    setAnimationPhase(phase);
  }, [lastProgressPhase]);

  return {
    // State
    progressState,
    isAnimating,
    collapsedSections,
    lastProgressPhase,
    animationPhase,
    currentPreMessage,
    revealedPostMessages,
    // Setters
    setProgressState,
    setIsAnimating,
    setCollapsedSections,
    setLastProgressPhase,
    setAnimationPhase,
    setCurrentPreMessage,
    setRevealedPostMessages,
    // Actions
    toggleSection,
    collapseSection,
    collapseSections,
    resetAnimationState,
    updatePhase,
  };
}

export default useProgress;

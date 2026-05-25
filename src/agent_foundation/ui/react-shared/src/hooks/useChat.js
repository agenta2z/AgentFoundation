/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Custom hook for managing chat state and messaging.
 * Handles messages, input, sending messages, and action handling.
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { API_BASE } from '../utils/api';
import { addDebugLog } from '../utils/debug';

/**
 * Hook for managing chat state and operations
 * @param {Function} startProgressPolling - Callback to start progress polling
 * @param {Function} resetProgressHeader - Callback to reset progress header state
 * @param {Function} handleExpandInputField - Callback to expand collapsible input fields
 * @returns {object} Chat state and handlers
 */
export function useChat(startProgressPolling, resetProgressHeader, handleExpandInputField) {
  // Chat messages
  const [messages, setMessages] = useState([]);
  // Input field value
  const [inputValue, setInputValue] = useState('');
  // Loading state
  const [isLoading, setIsLoading] = useState(false);
  // Suggested actions from backend
  const [suggestedActions, setSuggestedActions] = useState({ message: null, actions: [] });

  // Completed steps - each step contains user_message, post_messages, AND progress sections
  const [completedSteps, setCompletedSteps] = useState([]);

  // Pending user message - tracks the user message that triggered current animation
  const [pendingUserMessage, setPendingUserMessage] = useState(null);
  const pendingUserMessageRef = useRef(null);

  // Track when completion has been handled to avoid duplicate processing
  const [completionHandled, setCompletionHandled] = useState(false);
  const completionHandledRef = useRef(false);

  // Base message count before animation
  const [baseMessageCount, setBaseMessageCount] = useState(0);
  const baseMessageCountRef = useRef(0);

  // Debug: Log whenever completedSteps changes
  useEffect(() => {
    addDebugLog('state', 'completedSteps changed', {
      count: completedSteps.length,
      steps: completedSteps.map(s => ({
        step_id: s.step_id,
        post_messages_count: s.post_messages?.length || 0,
        progress_sections_count: s.progress_sections?.length || 0,
      })),
    });
  }, [completedSteps]);

  // Send message to backend
  const sendMessage = useCallback(async (message, callbacks) => {
    const {
      setProgressState,
      setIsAnimating,
      setCurrentPreMessage,
      setRevealedPostMessages,
      setAnimationPhase,
    } = callbacks;

    if (!message.trim() || isLoading) return;

    const trimmedMessage = message.trim();

    // Track this user message as pending
    setPendingUserMessage({
      role: 'user',
      content: trimmedMessage,
      message_type: 'text',
    });
    pendingUserMessageRef.current = {
      role: 'user',
      content: trimmedMessage,
      message_type: 'text',
    };

    setIsLoading(true);
    setInputValue('');

    // Reset animation state for new interactions
    setCompletionHandled(false);
    completionHandledRef.current = false;
    setProgressState(null);
    setCurrentPreMessage(null);
    setRevealedPostMessages([]);
    setAnimationPhase('idle');
    resetProgressHeader?.();

    try {
      const response = await fetch(`${API_BASE}/chat/send`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: trimmedMessage }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.is_animating) {
        // Track base message count
        const newBaseCount = messages.length + 1;
        setBaseMessageCount(newBaseCount);
        baseMessageCountRef.current = newBaseCount;

        setIsAnimating(true);
        startProgressPolling();

        if (data.suggested_actions?.actions?.length > 0) {
          setSuggestedActions(data.suggested_actions);
        }
      } else {
        // No animation - update messages normally
        setPendingUserMessage(null);
        pendingUserMessageRef.current = null;
        setMessages(data.messages);
        setSuggestedActions(data.suggested_actions || { message: null, actions: [] });
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setPendingUserMessage(null);
      pendingUserMessageRef.current = null;
      setMessages(prev => [...prev, {
        role: 'user',
        content: trimmedMessage,
        message_type: 'text',
      }, {
        role: 'assistant',
        content: `Error: ${error.message}. Please try again.`,
        message_type: 'text',
      }]);
    } finally {
      setIsLoading(false);
    }
  }, [isLoading, messages.length, startProgressPolling, resetProgressHeader]);

  // Handle suggested action click
  const handleAction = useCallback(async (actionIndex, callbacks) => {
    const {
      setProgressState,
      setIsAnimating,
      setCurrentPreMessage,
      setRevealedPostMessages,
      setAnimationPhase,
    } = callbacks;

    const action = suggestedActions.actions[actionIndex];

    // Handle expand_input action type
    if (action?.action_type === 'expand_input' && action.target_variable) {
      handleExpandInputField?.(action.target_variable);
      return;
    }

    // Track user message if present
    if (action?.user_message) {
      setPendingUserMessage({
        role: 'user',
        content: action.user_message,
        message_type: 'text',
      });
      pendingUserMessageRef.current = {
        role: 'user',
        content: action.user_message,
        message_type: 'text',
      };
    }

    setIsLoading(true);

    // Reset animation state
    setCompletionHandled(false);
    completionHandledRef.current = false;
    setProgressState(null);
    setCurrentPreMessage(null);
    setRevealedPostMessages([]);
    setAnimationPhase('idle');
    setSuggestedActions({ message: null, actions: [] });
    resetProgressHeader?.();

    try {
      const response = await fetch(`${API_BASE}/chat/action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ index: actionIndex }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.is_animating) {
        const currentMsgCount = messages.length;
        setBaseMessageCount(currentMsgCount);
        baseMessageCountRef.current = currentMsgCount;

        setIsAnimating(true);
        startProgressPolling();

        if (data.suggested_actions?.actions?.length > 0) {
          setSuggestedActions(data.suggested_actions);
        }
      } else {
        setMessages(data.messages);
        setSuggestedActions(data.suggested_actions || { message: null, actions: [] });
      }
    } catch (error) {
      console.error('Error handling action:', error);
    } finally {
      setIsLoading(false);
    }
  }, [messages.length, suggestedActions.actions, startProgressPolling, resetProgressHeader, handleExpandInputField]);

  // Handle branch action with user input
  const handleBranchAction = useCallback(async (actionIndex, inputValue, callbacks) => {
    const {
      setProgressState,
      setIsAnimating,
      setCurrentPreMessage,
      setRevealedPostMessages,
      setAnimationPhase,
    } = callbacks;

    const action = suggestedActions.actions[actionIndex];

    // Create user message showing the deep dive request
    const userMessage = `🔍 Deep dive on: ${inputValue}`;
    setPendingUserMessage({
      role: 'user',
      content: userMessage,
      message_type: 'text',
    });
    pendingUserMessageRef.current = {
      role: 'user',
      content: userMessage,
      message_type: 'text',
    };

    setIsLoading(true);

    // Reset animation state
    setCompletionHandled(false);
    completionHandledRef.current = false;
    setProgressState(null);
    setCurrentPreMessage(null);
    setRevealedPostMessages([]);
    setAnimationPhase('idle');
    setSuggestedActions({ message: null, actions: [] });
    resetProgressHeader?.();

    try {
      // Send branch action with input value
      const response = await fetch(`${API_BASE}/chat/branch_action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          index: actionIndex,
          input_value: inputValue,
          target_step: action?.target_step,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.is_animating) {
        const currentMsgCount = messages.length;
        setBaseMessageCount(currentMsgCount);
        baseMessageCountRef.current = currentMsgCount;

        setIsAnimating(true);
        startProgressPolling();

        if (data.suggested_actions?.actions?.length > 0) {
          setSuggestedActions(data.suggested_actions);
        }
      } else {
        setMessages(data.messages);
        setSuggestedActions(data.suggested_actions || { message: null, actions: [] });
      }
    } catch (error) {
      console.error('Error handling branch action:', error);
    } finally {
      setIsLoading(false);
    }
  }, [messages.length, suggestedActions.actions, startProgressPolling, resetProgressHeader]);

  // Clear pending user message
  const clearPendingUserMessage = useCallback(() => {
    setPendingUserMessage(null);
    pendingUserMessageRef.current = null;
  }, []);

  // Mark completion as handled
  const markCompletionHandled = useCallback(() => {
    setCompletionHandled(true);
    completionHandledRef.current = true;
  }, []);

  // Reset completion handling
  const resetCompletionHandled = useCallback(() => {
    setCompletionHandled(false);
    completionHandledRef.current = false;
  }, []);

  return {
    // State
    messages,
    inputValue,
    isLoading,
    suggestedActions,
    completedSteps,
    pendingUserMessage,
    completionHandled,
    baseMessageCount,
    // Refs
    pendingUserMessageRef,
    completionHandledRef,
    baseMessageCountRef,
    // Setters
    setMessages,
    setInputValue,
    setIsLoading,
    setSuggestedActions,
    setCompletedSteps,
    setPendingUserMessage,
    setCompletionHandled,
    setBaseMessageCount,
    // Actions
    sendMessage,
    handleAction,
    handleBranchAction,
    clearPendingUserMessage,
    markCompletionHandled,
    resetCompletionHandled,
  };
}

export default useChat;

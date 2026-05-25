/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Custom hook for managing user input fields state.
 * Handles input values and collapsible field states.
 */

import { useState, useCallback } from 'react';

/**
 * Hook for managing user input fields
 * @returns {object} Input fields state and handlers
 */
export function useInputFields() {
  // User input fields state (for input_field in post_messages)
  const [userInputFields, setUserInputFields] = useState({});

  // Collapsible input fields state - tracks which input fields are collapsed
  const [collapsedInputFields, setCollapsedInputFields] = useState({});

  const handleUserInputChange = useCallback((variableName, value) => {
    setUserInputFields(prev => ({
      ...prev,
      [variableName]: value,
    }));
  }, []);

  const handleExpandInputField = useCallback((variableName) => {
    setCollapsedInputFields(prev => ({
      ...prev,
      [variableName]: false,
    }));
  }, []);

  const getInputFieldValue = useCallback((variableName, defaultValue = '') => {
    return userInputFields[variableName] !== undefined
      ? userInputFields[variableName]
      : defaultValue;
  }, [userInputFields]);

  const isInputFieldCollapsed = useCallback((variableName, initiallyCollapsed = false) => {
    return collapsedInputFields[variableName] !== undefined
      ? collapsedInputFields[variableName]
      : initiallyCollapsed;
  }, [collapsedInputFields]);

  const resetInputFields = useCallback(() => {
    setUserInputFields({});
    setCollapsedInputFields({});
  }, []);

  return {
    // State
    userInputFields,
    collapsedInputFields,
    // Actions
    handleUserInputChange,
    handleExpandInputField,
    getInputFieldValue,
    isInputFieldCollapsed,
    resetInputFields,
  };
}

export default useInputFields;

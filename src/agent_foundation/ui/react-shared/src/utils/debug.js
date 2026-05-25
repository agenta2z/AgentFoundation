/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Debug utilities for the chatbot demo application.
 * Provides logging, filtering, and console access to debug information.
 */

// Global debug logs array for easy access from console
// Access with: window.debugLogs or copy with: window.copyDebugLogs()
// Filter by type with: window.filterDebugLogs('completion') or window.filterDebugLogs(['completion', 'state'])
export const initDebugTools = () => {
  if (typeof window === 'undefined') return;

  window.debugLogs = [];

  window.copyDebugLogs = (types) => {
    const logs = types ? window.filterDebugLogs(types) : window.debugLogs;
    const logsStr = JSON.stringify(logs, null, 2);
    navigator.clipboard.writeText(logsStr).then(() => {
      console.log(`Debug logs copied to clipboard! (${logs.length} entries)`);
    }).catch(() => {
      console.log('Copy failed. Here are the logs:\n', logsStr);
    });
    return logsStr;
  };

  window.clearDebugLogs = () => {
    window.debugLogs = [];
    console.log('Debug logs cleared');
  };

  window.filterDebugLogs = (types) => {
    const typeArray = Array.isArray(types) ? types : [types];
    return window.debugLogs.filter(log => typeArray.includes(log.type));
  };

  window.getDebugLogTypes = () => {
    const types = [...new Set(window.debugLogs.map(log => log.type))];
    console.log('Available log types:', types);
    return types;
  };
};

/**
 * Add a debug log entry with type categorization
 * @param {string} type - Log type for filtering: 'state', 'completion', 'polling', 'action', 'animation', 'progress_header', 'auto-advance'
 * @param {string} label - Human-readable label
 * @param {object} data - Data payload
 */
export const addDebugLog = (type, label, data) => {
  const entry = {
    timestamp: new Date().toISOString(),
    type,
    label,
    data,
  };
  console.log(`[DEBUG:${type}] ${label}:`, data);
  if (typeof window !== 'undefined' && window.debugLogs) {
    window.debugLogs.push(entry);
  }
};

/**
 * Create a showState function that can be attached to window for debugging
 * @param {Function} getStateSnapshot - Function that returns current state snapshot
 * @returns {Function} The showState function
 */
export const createShowState = (getStateSnapshot) => {
  return () => {
    const state = getStateSnapshot();
    console.log('[DEBUG] Current State:', state);
    return state;
  };
};

// Initialize debug tools on module load
initDebugTools();

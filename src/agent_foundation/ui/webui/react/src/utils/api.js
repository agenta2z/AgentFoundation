/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * API utilities for the chatbot demo application.
 * Centralized API configuration and helper functions.
 */

// API base URL (uses proxy in development)
export const API_BASE = '/api';

/**
 * Fetch JSON data from an API endpoint
 * @param {string} endpoint - API endpoint (without base URL)
 * @param {RequestInit} options - Fetch options
 * @returns {Promise<any>} Parsed JSON response
 */
export const fetchJson = async (endpoint, options = {}) => {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
};

/**
 * Send a POST request with JSON body
 * @param {string} endpoint - API endpoint (without base URL)
 * @param {object} data - Request body data
 * @returns {Promise<any>} Parsed JSON response
 */
export const postJson = async (endpoint, data) => {
  return fetchJson(endpoint, {
    method: 'POST',
    body: JSON.stringify(data),
  });
};

/**
 * Fetch text content from an API endpoint
 * @param {string} endpoint - API endpoint (without base URL)
 * @returns {Promise<string>} Text response
 */
export const fetchText = async (endpoint) => {
  const response = await fetch(`${API_BASE}${endpoint}`);

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.text();
};

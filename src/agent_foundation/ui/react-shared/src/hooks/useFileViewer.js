/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Custom hook for managing the file viewer drawer state.
 * Handles opening files, loading content, and managing HTML file preview.
 */

import { useState, useCallback } from 'react';
import { API_BASE } from '../utils/api';

/**
 * Hook for managing file viewer state and operations
 * @returns {object} File viewer state and handlers
 */
export function useFileViewer() {
  const [fileViewerOpen, setFileViewerOpen] = useState(false);
  const [fileContent, setFileContent] = useState('');
  const [fileName, setFileName] = useState('');
  const [isHtmlFile, setIsHtmlFile] = useState(false);
  const [htmlFilePath, setHtmlFilePath] = useState('');

  const openFileViewer = useCallback(async (filePath) => {
    try {
      const name = filePath.split('/').pop();
      const isHtml = name.endsWith('.html') || name.endsWith('.htm');

      if (isHtml) {
        // For HTML files, use the static-html endpoint directly via iframe src
        // This allows proper navigation between HTML pages
        setFileName(name);
        setIsHtmlFile(true);
        setHtmlFilePath(filePath);
        setFileContent(''); // Not needed for src-based iframe
        setFileViewerOpen(true);
      } else {
        // For non-HTML files, fetch the content as text
        const response = await fetch(`${API_BASE}/experiment/files/${encodeURIComponent(filePath)}`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const content = await response.text();
        setFileContent(content);
        setFileName(name);
        setIsHtmlFile(false);
        setHtmlFilePath('');
        setFileViewerOpen(true);
      }
    } catch (error) {
      console.error('Error loading file:', error);
    }
  }, []);

  const closeFileViewer = useCallback(() => {
    setFileViewerOpen(false);
  }, []);

  return {
    // State
    fileViewerOpen,
    fileContent,
    fileName,
    isHtmlFile,
    htmlFilePath,
    // Actions
    openFileViewer,
    closeFileViewer,
    setFileViewerOpen,
  };
}

export default useFileViewer;

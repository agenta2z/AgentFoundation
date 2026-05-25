/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Custom hook for managing context menu state.
 * Used for right-click menus on progress sections.
 */

import { useState, useCallback } from 'react';

/**
 * Hook for managing context menu state
 * @returns {object} Context menu state and handlers
 */
export function useContextMenu() {
  // Context menu state: { mouseX, mouseY, section }
  const [contextMenu, setContextMenu] = useState(null);

  const handleContextMenu = useCallback((event, section) => {
    event.preventDefault();
    setContextMenu({
      mouseX: event.clientX + 2,
      mouseY: event.clientY - 6,
      section: section,
    });
  }, []);

  const handleContextMenuClose = useCallback(() => {
    setContextMenu(null);
  }, []);

  return {
    contextMenu,
    handleContextMenu,
    handleContextMenuClose,
  };
}

export default useContextMenu;

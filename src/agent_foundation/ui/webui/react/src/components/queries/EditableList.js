/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * EditableList Component
 *
 * Generic container component that orchestrates QueryCard and AddDropdown.
 * Manages state for all items (selected, edited, added) and provides
 * onChange callback for parent components.
 *
 * This is a generic UI pattern component that can be used for:
 * - Research queries
 * - Proposals
 * - Any selectable list with additional items
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Box, Typography } from '@mui/material';
import QueryCard from './QueryCard';
import AddDropdown from './AddDropdown';

const EditableList = ({
  queries = [],
  additionalQueries = [],
  allowCustomQuery = true,
  variableName = 'selected_items',
  customQueryPlaceholder = 'Enter your own item...',
  addButtonLabel = 'Add More',
  customInputLabel = 'Custom Item',
  onChange,
}) => {
  // Debug: Log props to see what data is being passed
  console.log('[EditableList] Props received:', {
    queries,
    additionalQueries,
    allowCustomQuery,
    variableName,
    addButtonLabel,
    queriesLength: queries?.length,
  });

  // State for main queries (editable)
  const [mainQueries, setMainQueries] = useState(queries);

  // State for selected additional query IDs
  const [selectedAdditionalIds, setSelectedAdditionalIds] = useState([]);

  // State for custom queries added by user
  const [customQueries, setCustomQueries] = useState([]);

  // Initialize main queries when props change
  useEffect(() => {
    setMainQueries(queries);
  }, [queries]);

  // Notify parent of changes
  const notifyChange = useCallback((updatedMain, selectedAdditional, custom) => {
    if (onChange) {
      // Get selected additional queries
      const selectedAdditionalQueries = additionalQueries.filter(
        q => selectedAdditional.includes(q.id)
      );

      // Combine all selected queries
      const allSelectedQueries = {
        mainQueries: updatedMain,
        additionalQueries: selectedAdditionalQueries,
        customQueries: custom,
      };

      onChange(allSelectedQueries);
    }
  }, [onChange, additionalQueries]);

  // Handle main query changes
  const handleQueryChange = useCallback((updatedQuery) => {
    setMainQueries(prev => {
      const updated = prev.map(q =>
        q.id === updatedQuery.id ? updatedQuery : q
      );
      notifyChange(updated, selectedAdditionalIds, customQueries);
      return updated;
    });
  }, [selectedAdditionalIds, customQueries, notifyChange]);

  // Handle additional query selection
  const handleAdditionalQuerySelect = useCallback((selectedIds) => {
    setSelectedAdditionalIds(selectedIds);
    notifyChange(mainQueries, selectedIds, customQueries);
  }, [mainQueries, customQueries, notifyChange]);

  // Handle custom query addition
  const handleCustomQueryAdd = useCallback((queryText) => {
    const newCustomQuery = {
      id: `custom_${Date.now()}`,
      title: `Custom: ${queryText.slice(0, 50)}${queryText.length > 50 ? '...' : ''}`,
      focus: 'Custom Item',
      description: queryText,
      isCustom: true,
    };

    setCustomQueries(prev => {
      const updated = [...prev, newCustomQuery];
      notifyChange(mainQueries, selectedAdditionalIds, updated);
      return updated;
    });
  }, [mainQueries, selectedAdditionalIds, notifyChange]);

  // Handle custom query removal
  const handleCustomQueryRemove = useCallback((queryId) => {
    setCustomQueries(prev => {
      const updated = prev.filter(q => q.id !== queryId);
      notifyChange(mainQueries, selectedAdditionalIds, updated);
      return updated;
    });
  }, [mainQueries, selectedAdditionalIds, notifyChange]);

  return (
    <Box
      className="editable-query-list"
      sx={{
        mt: 2,
        width: '100%',
      }}
    >
      {/* Main Queries */}
      {mainQueries.map((query, index) => (
        <QueryCard
          key={query.id || index}
          query={query}
          onQueryChange={handleQueryChange}
          isEditable={true}
        />
      ))}

      {/* Custom Queries (shown as editable cards) */}
      {customQueries.map((query, index) => (
        <Box key={query.id} sx={{ position: 'relative' }}>
          <QueryCard
            query={{
              ...query,
              rank: '✏️',
            }}
            onQueryChange={(updatedQuery) => {
              setCustomQueries(prev => {
                const updated = prev.map(q =>
                  q.id === updatedQuery.id ? updatedQuery : q
                );
                notifyChange(mainQueries, selectedAdditionalIds, updated);
                return updated;
              });
            }}
            isEditable={true}
          />
          {/* Remove button for custom queries */}
          <Typography
            component="span"
            onClick={() => handleCustomQueryRemove(query.id)}
            sx={{
              position: 'absolute',
              top: 8,
              right: 8,
              cursor: 'pointer',
              color: 'text.secondary',
              fontSize: '0.85rem',
              opacity: 0.7,
              '&:hover': {
                opacity: 1,
                color: 'error.main',
              },
            }}
          >
            ✕ Remove
          </Typography>
        </Box>
      ))}

      {/* Add More Dropdown */}
      <AddDropdown
        additionalItems={additionalQueries}
        selectedIds={selectedAdditionalIds}
        onSelect={handleAdditionalQuerySelect}
        allowCustom={allowCustomQuery}
        customPlaceholder={customQueryPlaceholder}
        customInputLabel={customInputLabel}
        onCustomAdd={handleCustomQueryAdd}
        buttonLabel={addButtonLabel}
      />
    </Box>
  );
};

export default EditableList;

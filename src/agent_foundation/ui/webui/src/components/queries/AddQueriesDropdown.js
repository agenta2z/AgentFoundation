/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * AddQueriesDropdown Component
 *
 * A collapsible section for adding additional queries from a predefined list
 * or creating custom queries. Features checkbox selection and custom input.
 */

import React, { useState } from 'react';
import {
  Box,
  Typography,
  Checkbox,
  FormControlLabel,
  TextField,
  Collapse,
  IconButton,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Add as AddIcon,
} from '@mui/icons-material';

const AddQueriesDropdown = ({
  additionalQueries = [],
  selectedQueryIds = [],
  onQuerySelect,
  allowCustomQuery = true,
  customQueryPlaceholder = 'Enter your own research query...',
  onCustomQueryAdd,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [customQueryText, setCustomQueryText] = useState('');

  const handleToggleExpand = () => {
    setIsExpanded(!isExpanded);
  };

  const handleCheckboxChange = (queryId) => {
    if (onQuerySelect) {
      const isCurrentlySelected = selectedQueryIds.includes(queryId);
      if (isCurrentlySelected) {
        onQuerySelect(selectedQueryIds.filter(id => id !== queryId));
      } else {
        onQuerySelect([...selectedQueryIds, queryId]);
      }
    }
  };

  const handleCustomQuerySubmit = () => {
    if (customQueryText.trim() && onCustomQueryAdd) {
      onCustomQueryAdd(customQueryText.trim());
      setCustomQueryText('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleCustomQuerySubmit();
    }
  };

  return (
    <Box
      className="add-queries-section"
      sx={{
        backgroundColor: 'rgba(255, 255, 255, 0.02)',
        border: '1px dashed rgba(255, 255, 255, 0.15)',
        borderRadius: '8px',
        marginTop: '16px',
        overflow: 'hidden',
      }}
    >
      {/* Dropdown Header */}
      <Box
        onClick={handleToggleExpand}
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '12px 16px',
          cursor: 'pointer',
          transition: 'background-color 0.2s ease',
          '&:hover': {
            backgroundColor: 'rgba(255, 255, 255, 0.03)',
          },
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <AddIcon sx={{ fontSize: 20, color: 'primary.main' }} />
          <Typography
            sx={{
              fontWeight: 500,
              color: 'text.primary',
              fontSize: '0.95rem',
            }}
          >
            Add More Queries
          </Typography>
          {selectedQueryIds.length > 0 && (
            <Typography
              component="span"
              sx={{
                backgroundColor: 'primary.main',
                color: 'white',
                borderRadius: '12px',
                padding: '2px 8px',
                fontSize: '0.75rem',
                fontWeight: 600,
              }}
            >
              {selectedQueryIds.length} selected
            </Typography>
          )}
        </Box>
        <Box sx={{ color: 'text.secondary' }}>
          {isExpanded ? (
            <ExpandLessIcon sx={{ fontSize: 20 }} />
          ) : (
            <ExpandMoreIcon sx={{ fontSize: 20 }} />
          )}
        </Box>
      </Box>

      {/* Expandable Content */}
      <Collapse in={isExpanded} timeout="auto">
        <Box
          sx={{
            padding: '0 16px 16px',
            borderTop: '1px solid rgba(255, 255, 255, 0.08)',
          }}
        >
          {/* Additional Queries List */}
          {additionalQueries.length > 0 && (
            <Box sx={{ mt: 2 }}>
              {additionalQueries.map((query) => (
                <Box
                  key={query.id}
                  className="additional-query-item"
                  sx={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    padding: '8px 0',
                    borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                    '&:last-child': {
                      borderBottom: 'none',
                    },
                  }}
                >
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={selectedQueryIds.includes(query.id)}
                        onChange={() => handleCheckboxChange(query.id)}
                        sx={{
                          color: 'rgba(255, 255, 255, 0.3)',
                          '&.Mui-checked': {
                            color: 'primary.main',
                          },
                        }}
                      />
                    }
                    label={
                      <Box>
                        <Typography
                          sx={{
                            color: 'text.primary',
                            fontSize: '0.9rem',
                            fontWeight: 500,
                          }}
                        >
                          {query.title}
                        </Typography>
                        <Typography
                          sx={{
                            color: 'text.secondary',
                            fontSize: '0.8rem',
                            opacity: 0.8,
                          }}
                        >
                          {query.focus}
                        </Typography>
                        {query.description && (
                          <Typography
                            sx={{
                              color: 'text.secondary',
                              fontSize: '0.75rem',
                              opacity: 0.7,
                              marginTop: '4px',
                              lineHeight: 1.4,
                            }}
                          >
                            {query.description}
                          </Typography>
                        )}
                      </Box>
                    }
                    sx={{
                      alignItems: 'flex-start',
                      margin: 0,
                      width: '100%',
                      '& .MuiFormControlLabel-label': {
                        flex: 1,
                      },
                    }}
                  />
                </Box>
              ))}
            </Box>
          )}

          {/* Custom Query Input */}
          {allowCustomQuery && (
            <Box sx={{ mt: 2 }}>
              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1,
                  mb: 1,
                }}
              >
                <Typography
                  sx={{
                    color: 'text.secondary',
                    fontSize: '0.8rem',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                  }}
                >
                  ✏️ Custom Query
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <TextField
                  fullWidth
                  placeholder={customQueryPlaceholder}
                  value={customQueryText}
                  onChange={(e) => setCustomQueryText(e.target.value)}
                  onKeyDown={handleKeyDown}
                  variant="outlined"
                  size="small"
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      backgroundColor: 'rgba(0, 0, 0, 0.2)',
                      borderRadius: 1,
                      '& fieldset': { borderColor: 'rgba(255, 255, 255, 0.15)' },
                      '&:hover fieldset': { borderColor: 'rgba(255, 255, 255, 0.25)' },
                      '&.Mui-focused fieldset': { borderColor: 'primary.main' },
                    },
                    '& .MuiInputBase-input': {
                      color: 'text.primary',
                      fontSize: '0.9rem',
                    },
                  }}
                />
                <IconButton
                  onClick={handleCustomQuerySubmit}
                  disabled={!customQueryText.trim()}
                  sx={{
                    backgroundColor: 'primary.main',
                    color: 'white',
                    '&:hover': {
                      backgroundColor: 'primary.dark',
                    },
                    '&.Mui-disabled': {
                      backgroundColor: 'rgba(255, 255, 255, 0.1)',
                      color: 'rgba(255, 255, 255, 0.3)',
                    },
                  }}
                >
                  <AddIcon />
                </IconButton>
              </Box>
              <Typography
                variant="caption"
                sx={{
                  color: 'text.secondary',
                  mt: 1,
                  display: 'block',
                  opacity: 0.7,
                }}
              >
                Press Enter or click + to add your custom query
              </Typography>
            </Box>
          )}
        </Box>
      </Collapse>
    </Box>
  );
};

export default AddQueriesDropdown;

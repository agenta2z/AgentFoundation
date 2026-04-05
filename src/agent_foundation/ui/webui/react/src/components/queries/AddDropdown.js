/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * AddDropdown Component
 *
 * A generic collapsible section for adding additional items from a predefined list
 * or creating custom items. Features checkbox selection and custom input.
 * Can be used for queries, proposals, or any selectable list.
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
import { useTheme } from '@mui/material/styles';
import {
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Add as AddIcon,
} from '@mui/icons-material';

const AddDropdown = ({
  additionalItems = [],
  selectedIds = [],
  onSelect,
  allowCustom = true,
  customPlaceholder = 'Enter your own item...',
  customInputLabel = 'Custom Item',
  onCustomAdd,
  buttonLabel = 'Add More',
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [customText, setCustomText] = useState('');
  const theme = useTheme();

  // Don't render if there's nothing to add
  if (additionalItems.length === 0 && !allowCustom) {
    return null;
  }

  const handleToggleExpand = () => {
    setIsExpanded(!isExpanded);
  };

  const handleCheckboxChange = (itemId) => {
    if (onSelect) {
      const isCurrentlySelected = selectedIds.includes(itemId);
      if (isCurrentlySelected) {
        onSelect(selectedIds.filter(id => id !== itemId));
      } else {
        onSelect([...selectedIds, itemId]);
      }
    }
  };

  const handleCustomSubmit = () => {
    if (customText.trim() && onCustomAdd) {
      onCustomAdd(customText.trim());
      setCustomText('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleCustomSubmit();
    }
  };

  return (
    <Box
      className="add-dropdown-section"
      sx={{
        backgroundColor: theme.custom.surfaces.cardBg,
        border: `1px dashed ${theme.custom.surfaces.inputBorder}`,
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
            backgroundColor: theme.custom.surfaces.cardBg,
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
            {buttonLabel}
          </Typography>
          {selectedIds.length > 0 && (
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
              {selectedIds.length} selected
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
            borderTop: `1px solid ${theme.custom.surfaces.overlayMedium}`,
          }}
        >
          {/* Additional Items List */}
          {additionalItems.length > 0 && (
            <Box sx={{ mt: 2 }}>
              {additionalItems.map((item) => (
                <Box
                  key={item.id}
                  className="additional-item"
                  sx={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    padding: '8px 0',
                    borderBottom: `1px solid ${theme.custom.surfaces.inputBg}`,
                    '&:last-child': {
                      borderBottom: 'none',
                    },
                  }}
                >
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={selectedIds.includes(item.id)}
                        onChange={() => handleCheckboxChange(item.id)}
                        sx={{
                          color: theme.custom.surfaces.inputBorderHover,
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
                          {item.title}
                        </Typography>
                        {item.focus && (
                          <Typography
                            sx={{
                              color: 'text.secondary',
                              fontSize: '0.8rem',
                              opacity: 0.8,
                            }}
                          >
                            {item.focus}
                          </Typography>
                        )}
                        {item.description && (
                          <Typography
                            sx={{
                              color: 'text.secondary',
                              fontSize: '0.75rem',
                              opacity: 0.7,
                              marginTop: '4px',
                              lineHeight: 1.4,
                            }}
                          >
                            {item.description}
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

          {/* Custom Input */}
          {allowCustom && (
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
                  {customInputLabel}
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <TextField
                  fullWidth
                  placeholder={customPlaceholder}
                  value={customText}
                  onChange={(e) => setCustomText(e.target.value)}
                  onKeyDown={handleKeyDown}
                  variant="outlined"
                  size="small"
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      backgroundColor: theme.custom.surfaces.sidebarBg,
                      borderRadius: 1,
                      '& fieldset': { borderColor: theme.custom.surfaces.inputBorder },
                      '&:hover fieldset': { borderColor: theme.custom.surfaces.inputBorderHover },
                      '&.Mui-focused fieldset': { borderColor: 'primary.main' },
                    },
                    '& .MuiInputBase-input': {
                      color: 'text.primary',
                      fontSize: '0.9rem',
                    },
                  }}
                />
                <IconButton
                  onClick={handleCustomSubmit}
                  disabled={!customText.trim()}
                  sx={{
                    backgroundColor: 'primary.main',
                    color: 'white',
                    '&:hover': {
                      backgroundColor: 'primary.dark',
                    },
                    '&.Mui-disabled': {
                      backgroundColor: theme.custom.surfaces.overlayActive,
                      color: theme.custom.surfaces.inputBorderHover,
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
                Press Enter or click + to add
              </Typography>
            </Box>
          )}
        </Box>
      </Collapse>
    </Box>
  );
};

export default AddDropdown;

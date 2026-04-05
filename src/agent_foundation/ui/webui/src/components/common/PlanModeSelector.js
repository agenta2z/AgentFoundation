/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * PlanModeSelector component - Dropdown for selecting research mode
 * with editable textarea and draft preservation.
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  IconButton,
  Typography,
  Tooltip,
  CircularProgress,
} from '@mui/material';
import { Refresh as RefreshIcon } from '@mui/icons-material';
import { API_BASE } from '../../utils/api';

/**
 * PlanModeSelector - Dropdown + textarea component for research plan customization
 *
 * @param {object} props
 * @param {object} props.modeSelector - Mode selector configuration from flow.json
 * @param {string} props.variableName - Variable name for the input field
 * @param {string} props.placeholder - Placeholder text for textarea
 * @param {function} props.onChange - Callback when value changes (variableName, value)
 * @param {string} props.currentValue - Current value from parent state
 */
export function PlanModeSelector({
  modeSelector,
  variableName,
  placeholder,
  onChange,
  currentValue,
}) {
  const [selectedMode, setSelectedMode] = useState(modeSelector.default_mode);
  const [isLoading, setIsLoading] = useState(true);
  const [originalTemplates, setOriginalTemplates] = useState({});
  const [userDrafts, setUserDrafts] = useState({});
  const [isModified, setIsModified] = useState(false);

  // Load content file from backend
  const loadContentFile = useCallback(async (contentFile) => {
    try {
      const response = await fetch(`${API_BASE}/experiment/files/${encodeURIComponent(contentFile)}`);
      if (!response.ok) {
        throw new Error(`Failed to load content file: ${response.status}`);
      }
      const content = await response.text();
      return content;
    } catch (error) {
      console.error('Error loading content file:', error);
      return `# Error loading content\n\nCould not load content file: ${contentFile}`;
    }
  }, []);

  // Load initial content on mount
  useEffect(() => {
    const loadInitialContent = async () => {
      setIsLoading(true);
      const defaultOption = modeSelector.options.find(opt => opt.id === modeSelector.default_mode);
      if (defaultOption) {
        const content = await loadContentFile(defaultOption.file);
        setOriginalTemplates(prev => ({ ...prev, [defaultOption.id]: content }));
        onChange(variableName, content);
      }
      setIsLoading(false);
    };
    loadInitialContent();
  }, [modeSelector, loadContentFile, onChange, variableName]);

  // Check if current content is modified from original template
  useEffect(() => {
    const originalTemplate = originalTemplates[selectedMode];
    if (originalTemplate && currentValue) {
      setIsModified(currentValue !== originalTemplate);
    }
  }, [currentValue, selectedMode, originalTemplates]);

  // Handle mode change
  const handleModeChange = async (event) => {
    const newModeId = event.target.value;

    // Save current content as draft if modified
    if (isModified && currentValue) {
      setUserDrafts(prev => ({ ...prev, [selectedMode]: currentValue }));
    }

    setSelectedMode(newModeId);

    // Check if we have a user draft for this mode
    if (userDrafts[newModeId]) {
      onChange(variableName, userDrafts[newModeId]);
      return;
    }

    // Check if we have the original template cached
    if (originalTemplates[newModeId]) {
      onChange(variableName, originalTemplates[newModeId]);
      return;
    }

    // Load the content file for this mode
    setIsLoading(true);
    const option = modeSelector.options.find(opt => opt.id === newModeId);
    if (option) {
      const content = await loadContentFile(option.file);
      setOriginalTemplates(prev => ({ ...prev, [newModeId]: content }));
      onChange(variableName, content);
    }
    setIsLoading(false);
  };

  // Handle reset to original template
  const handleReset = () => {
    const originalTemplate = originalTemplates[selectedMode];
    if (originalTemplate) {
      onChange(variableName, originalTemplate);
      // Clear the user draft for this mode
      setUserDrafts(prev => {
        const newDrafts = { ...prev };
        delete newDrafts[selectedMode];
        return newDrafts;
      });
    }
  };

  // Handle textarea change
  const handleTextChange = (event) => {
    onChange(variableName, event.target.value);
  };

  const selectedOption = modeSelector.options.find(opt => opt.id === selectedMode);

  return (
    <Box sx={{ mt: 2 }}>
      {/* Mode Selector Dropdown */}
      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel
          id="research-mode-label"
          sx={{ color: 'text.secondary' }}
        >
          {modeSelector.label}
        </InputLabel>
        <Select
          labelId="research-mode-label"
          value={selectedMode}
          label={modeSelector.label}
          onChange={handleModeChange}
          disabled={isLoading}
          sx={{
            backgroundColor: 'rgba(0, 0, 0, 0.2)',
            '& .MuiOutlinedInput-notchedOutline': {
              borderColor: 'rgba(255, 255, 255, 0.2)',
            },
            '&:hover .MuiOutlinedInput-notchedOutline': {
              borderColor: 'rgba(255, 255, 255, 0.3)',
            },
            '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
              borderColor: 'primary.main',
            },
            '& .MuiSelect-select': {
              color: 'text.primary',
            },
          }}
        >
          {modeSelector.options.map((option) => (
            <MenuItem key={option.id} value={option.id}>
              <Box>
                <Typography variant="body1">{option.label}</Typography>
                <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                  {option.description}
                </Typography>
              </Box>
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {/* Textarea with Reset Button */}
      <Box sx={{ position: 'relative' }}>
        {isLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
            <CircularProgress size={24} />
          </Box>
        ) : (
          <>
            <TextField
              fullWidth
              multiline
              minRows={8}
              maxRows={20}
              placeholder={placeholder || 'Customize the research plan here...'}
              value={currentValue || ''}
              onChange={handleTextChange}
              variant="outlined"
              sx={{
                '& .MuiOutlinedInput-root': {
                  backgroundColor: 'rgba(0, 0, 0, 0.2)',
                  borderRadius: 1,
                  '& fieldset': { borderColor: 'rgba(255, 255, 255, 0.2)' },
                  '&:hover fieldset': { borderColor: 'rgba(255, 255, 255, 0.3)' },
                  '&.Mui-focused fieldset': { borderColor: 'primary.main' },
                },
                '& .MuiInputBase-input': {
                  color: 'text.primary',
                  fontSize: '0.9rem',
                  fontFamily: 'monospace',
                },
              }}
            />

            {/* Reset Button - only show when modified */}
            {isModified && (
              <Tooltip title="Reset to original template">
                <IconButton
                  onClick={handleReset}
                  size="small"
                  sx={{
                    position: 'absolute',
                    top: 8,
                    right: 8,
                    backgroundColor: 'rgba(0, 0, 0, 0.5)',
                    color: 'text.secondary',
                    '&:hover': {
                      backgroundColor: 'rgba(0, 0, 0, 0.7)',
                      color: 'primary.main',
                    },
                  }}
                >
                  <RefreshIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            )}
          </>
        )}
      </Box>

      {/* Modified Indicator */}
      {isModified && (
        <Typography
          variant="caption"
          sx={{ color: 'warning.main', mt: 0.5, display: 'block' }}
        >
          ✏️ Modified from template - click ↺ to reset
        </Typography>
      )}

      <Typography
        variant="caption"
        sx={{ color: 'text.secondary', mt: 0.5, display: 'block' }}
      >
        Optional - you can leave this as-is or customize
      </Typography>
    </Box>
  );
}

export default PlanModeSelector;

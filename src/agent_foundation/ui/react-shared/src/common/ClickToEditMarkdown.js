/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * ClickToEditMarkdown Component
 *
 * A component that displays rendered markdown by default,
 * and switches to an editable text field when clicked.
 * Supports both view mode (rendered markdown) and edit mode (TextField).
 */

import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Typography,
  TextField,
} from '@mui/material';
import {
  Edit as EditIcon,
} from '@mui/icons-material';
import { MarkdownRenderer } from './MarkdownRenderer';

const ClickToEditMarkdown = ({
  value,
  onChange,
  placeholder = 'Click to edit...',
  minRows = 3,
  maxRows = 12,
  helperText,
  disabled = false,
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const textFieldRef = useRef(null);

  // Focus the text field when entering edit mode
  useEffect(() => {
    if (isEditing && textFieldRef.current) {
      textFieldRef.current.focus();
      // Move cursor to end
      const length = textFieldRef.current.value?.length || 0;
      textFieldRef.current.setSelectionRange(length, length);
    }
  }, [isEditing]);

  const handleMarkdownClick = () => {
    if (!disabled) {
      setIsEditing(true);
    }
  };

  const handleTextFieldBlur = () => {
    setIsEditing(false);
  };

  const handleKeyDown = (e) => {
    // Exit edit mode on Escape
    if (e.key === 'Escape') {
      setIsEditing(false);
    }
  };

  const handleChange = (e) => {
    if (onChange) {
      onChange(e.target.value);
    }
  };

  if (isEditing) {
    // Edit mode: Show TextField
    return (
      <Box>
        <TextField
          fullWidth
          multiline
          minRows={minRows}
          maxRows={maxRows}
          value={value}
          onChange={handleChange}
          onBlur={handleTextFieldBlur}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          variant="outlined"
          inputRef={textFieldRef}
          sx={(muiTheme) => ({
            '& .MuiOutlinedInput-root': {
              backgroundColor: muiTheme.custom.surfaces.inputBg,
              borderRadius: 1,
              '& fieldset': { borderColor: 'primary.main' },
              '&:hover fieldset': { borderColor: 'primary.light' },
              '&.Mui-focused fieldset': { borderColor: 'primary.main' },
            },
            '& .MuiInputBase-input': {
              color: 'text.primary',
              fontSize: '0.9rem',
              lineHeight: 1.6,
              fontFamily: 'monospace',
            },
          }}
        />
        <Typography
          variant="caption"
          sx={{
            color: 'text.secondary',
            mt: 1,
            display: 'block',
            opacity: 0.7,
          }}
        >
          💡 Markdown supported • Press Escape or click outside to preview
        </Typography>
        {helperText && (
          <Typography variant="caption" sx={{ color: 'text.secondary', mt: 0.5, display: 'block' }}>
            {helperText}
          </Typography>
        )}
      </Box>
    );
  }

  // View mode: Show rendered markdown (click to edit)
  return (
    <Box>
      <Box
        onClick={handleMarkdownClick}
        sx={(muiTheme) => ({
          backgroundColor: muiTheme.custom.surfaces.inputBg,
          borderRadius: 1,
          padding: '12px 16px',
          minHeight: '80px',
          maxHeight: '300px',
          overflowY: 'auto',
          cursor: disabled ? 'default' : 'text',
          border: '1px solid',
          borderColor: 'divider',
          transition: 'all 0.2s ease',
          position: 'relative',
          '&:hover': !disabled ? {
            backgroundColor: muiTheme.custom.surfaces.overlayMedium,
            borderColor: muiTheme.custom.surfaces.highlightBorder,
          } : {},
          // Markdown content styling
          '& p': { margin: '0 0 8px 0' },
          '& p:last-child': { marginBottom: 0 },
          '& ul, & ol': { margin: '8px 0', paddingLeft: '20px' },
          '& li': { marginBottom: '4px' },
          '& code': {
            backgroundColor: muiTheme.custom.surfaces.overlayMedium,
            padding: '2px 6px',
            borderRadius: '4px',
            fontSize: '0.85em',
          },
          '& pre': {
            margin: '8px 0',
            overflow: 'auto',
          },
          '& h1, & h2, & h3, & h4': {
            margin: '12px 0 8px 0',
            fontWeight: 600,
          },
          '& table': {
            margin: '8px 0',
            borderCollapse: 'collapse',
            '& th, & td': {
              border: `1px solid ${muiTheme.custom.surfaces.overlayMedium}`,
              padding: '6px 10px',
            },
          },
          '& blockquote': {
            borderLeft: `3px solid ${muiTheme.custom.surfaces.highlightBorder}`,
            margin: '8px 0',
            paddingLeft: '12px',
            color: 'text.secondary',
          },
          '& hr': {
            border: 'none',
            borderTop: `1px solid ${muiTheme.custom.surfaces.overlayMedium}`,
            margin: '16px 0',
          },
          '& strong': {
            fontWeight: 600,
          },
          '& em': {
            fontStyle: 'italic',
          },
        })}
      >
        {/* Click to edit hint */}
        {!disabled && (
          <Box
            sx={{
              position: 'absolute',
              top: 8,
              right: 8,
              display: 'flex',
              alignItems: 'center',
              gap: 0.5,
              color: 'primary.main',
              fontSize: '0.7rem',
              opacity: 0.7,
              transition: 'opacity 0.2s',
              '&:hover': {
                opacity: 1,
              },
            }}
          >
            <EditIcon sx={{ fontSize: 12 }} />
            <Typography variant="caption" sx={{ fontSize: '0.7rem' }}>
              Click to edit
            </Typography>
          </Box>
        )}

        {value ? (
          <MarkdownRenderer content={value} />
        ) : (
          <Typography
            sx={{
              color: 'text.secondary',
              fontStyle: 'italic',
              opacity: 0.6,
            }}
          >
            {disabled ? 'No content' : placeholder}
          </Typography>
        )}
      </Box>
      {helperText && (
        <Typography variant="caption" sx={{ color: 'text.secondary', mt: 0.5, display: 'block' }}>
          {helperText}
        </Typography>
      )}
    </Box>
  );
};

export { ClickToEditMarkdown };
export default ClickToEditMarkdown;

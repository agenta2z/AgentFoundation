/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * QueryCard Component
 *
 * Displays a single research query with rank emoji, title, and focus area.
 * Features click-to-edit markdown: shows rendered markdown by default,
 * click on markdown area to switch to editable text mode.
 */

import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Typography,
  TextField,
  Collapse,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import {
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Edit as EditIcon,
} from '@mui/icons-material';
import { MarkdownRenderer } from '../common/MarkdownRenderer';

const QueryCard = ({
  query,
  onQueryChange,
  isEditable = true,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editedDescription, setEditedDescription] = useState(query.description || '');
  const textFieldRef = useRef(null);
  const theme = useTheme();

  // Focus the text field when entering edit mode
  useEffect(() => {
    if (isEditing && textFieldRef.current) {
      textFieldRef.current.focus();
    }
  }, [isEditing]);

  // Update local state when query prop changes
  useEffect(() => {
    setEditedDescription(query.description || '');
  }, [query.description]);

  const handleToggleExpand = () => {
    if (isEditable) {
      setIsExpanded(!isExpanded);
      // Exit edit mode when collapsing
      if (isExpanded) {
        setIsEditing(false);
      }
    }
  };

  const handleDescriptionChange = (e) => {
    const newDescription = e.target.value;
    setEditedDescription(newDescription);
    if (onQueryChange) {
      onQueryChange({
        ...query,
        description: newDescription,
      });
    }
  };

  const handleMarkdownClick = (e) => {
    e.stopPropagation();
    if (isEditable) {
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

  return (
    <Box
      className={`query-card ${isExpanded ? 'expanded' : ''}`}
      sx={{
        backgroundColor: isExpanded
          ? theme.custom.surfaces.overlayMedium
          : theme.custom.surfaces.inputBg,
        borderRadius: '8px',
        marginBottom: '8px',
        cursor: isEditable ? 'pointer' : 'default',
        transition: 'all 0.2s ease',
        border: '1px solid',
        borderColor: isExpanded
          ? theme.custom.surfaces.highlightBorder
          : theme.custom.surfaces.overlayActive,
        '&:hover': {
          backgroundColor: isEditable
            ? theme.custom.surfaces.overlayMedium
            : theme.custom.surfaces.inputBg,
          borderColor: isEditable
            ? theme.custom.surfaces.highlight
            : theme.custom.surfaces.overlayActive,
        },
      }}
    >
      {/* Card Header */}
      <Box
        onClick={handleToggleExpand}
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '12px 16px',
        }}
      >
        <Box sx={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography
              component="span"
              sx={{ fontSize: '1.1rem' }}
            >
              {query.rank}
            </Typography>
            <Typography
              component="span"
              sx={{
                fontWeight: 600,
                color: 'text.primary',
                fontSize: '0.95rem',
              }}
            >
              {query.title}
            </Typography>
          </Box>
          <Typography
            variant="body2"
            sx={{
              color: 'text.secondary',
              mt: 0.5,
              fontSize: '0.85rem',
              opacity: 0.8,
            }}
          >
            {query.focus}
          </Typography>
        </Box>

        {isEditable && (
          <Box sx={{
            display: 'flex',
            alignItems: 'center',
            color: 'text.secondary',
            ml: 2,
          }}>
            {isExpanded ? (
              <ExpandLessIcon sx={{ fontSize: 20 }} />
            ) : (
              <>
                <Typography
                  variant="caption"
                  sx={{
                    mr: 0.5,
                    opacity: 0.7,
                    fontSize: '0.75rem',
                  }}
                >
                  Click to expand
                </Typography>
                <ExpandMoreIcon sx={{ fontSize: 20 }} />
              </>
            )}
          </Box>
        )}
      </Box>

      {/* Expandable Content */}
      <Collapse in={isExpanded} timeout="auto">
        <Box
          sx={{
            padding: '0 16px 16px',
            borderTop: `1px solid ${theme.custom.surfaces.overlayActive}`,
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              mt: 2,
              mb: 1,
            }}
          >
            <Typography
              variant="subtitle2"
              sx={{
                color: 'text.secondary',
                fontSize: '0.8rem',
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
              }}
            >
              {isEditing ? '✏️ Editing Description' : 'Description'}
            </Typography>
            {!isEditing && isEditable && (
              <Typography
                variant="caption"
                sx={{
                  color: 'primary.main',
                  fontSize: '0.7rem',
                  opacity: 0.8,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 0.5,
                }}
              >
                <EditIcon sx={{ fontSize: 12 }} />
                Click to edit
              </Typography>
            )}
          </Box>

          {/* Click-to-edit markdown content */}
          {isEditing ? (
            // Edit mode: Show TextField
            <TextField
              fullWidth
              multiline
              minRows={3}
              maxRows={12}
              value={editedDescription}
              onChange={handleDescriptionChange}
              onBlur={handleTextFieldBlur}
              onKeyDown={handleKeyDown}
              placeholder="Edit the research query description (Markdown supported)..."
              variant="outlined"
              inputRef={textFieldRef}
              sx={{
                '& .MuiOutlinedInput-root': {
                  backgroundColor: theme.custom.surfaces.sidebarBg,
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
          ) : (
            // View mode: Show rendered markdown (click to edit)
            <Box
              onClick={handleMarkdownClick}
              sx={{
                backgroundColor: theme.custom.surfaces.sidebarBg,
                borderRadius: 1,
                padding: '12px 16px',
                minHeight: '80px',
                cursor: isEditable ? 'text' : 'default',
                border: `1px solid ${theme.custom.surfaces.overlayActive}`,
                transition: 'all 0.2s ease',
                '&:hover': isEditable ? {
                  backgroundColor: theme.custom.surfaces.sidebarBg,
                  borderColor: theme.custom.surfaces.highlightBorder,
                } : {},
                // Markdown content styling
                '& p': { margin: '0 0 8px 0' },
                '& p:last-child': { marginBottom: 0 },
                '& ul, & ol': { margin: '8px 0', paddingLeft: '20px' },
                '& li': { marginBottom: '4px' },
                '& code': {
                  backgroundColor: theme.custom.surfaces.scrim,
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
                },
                '& blockquote': {
                  borderLeft: `3px solid ${theme.custom.surfaces.mutedText}`,
                  margin: '8px 0',
                  paddingLeft: '12px',
                  color: 'text.secondary',
                },
              }}
            >
              {editedDescription ? (
                <MarkdownRenderer content={editedDescription} />
              ) : (
                <Typography
                  sx={{
                    color: 'text.secondary',
                    fontStyle: 'italic',
                    opacity: 0.6,
                  }}
                >
                  {isEditable ? 'Click to add description...' : 'No description provided'}
                </Typography>
              )}
            </Box>
          )}

          {/* Help text when editing */}
          {isEditing && (
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
          )}
        </Box>
      </Collapse>
    </Box>
  );
};

export default QueryCard;

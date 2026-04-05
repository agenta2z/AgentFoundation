/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * QueryCard Component
 *
 * Displays a single research query with rank emoji, title, and focus area.
 * Click to expand/collapse for inline editing of the description.
 */

import React, { useState } from 'react';
import {
  Box,
  Typography,
  TextField,
  Collapse,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
} from '@mui/icons-material';

const QueryCard = ({
  query,
  onQueryChange,
  isEditable = true,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [editedDescription, setEditedDescription] = useState(query.description || '');

  const handleToggleExpand = () => {
    if (isEditable) {
      setIsExpanded(!isExpanded);
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

  return (
    <Box
      className={`query-card ${isExpanded ? 'expanded' : ''}`}
      sx={{
        backgroundColor: isExpanded
          ? 'rgba(255, 255, 255, 0.07)'
          : 'rgba(255, 255, 255, 0.05)',
        borderRadius: '8px',
        marginBottom: '8px',
        cursor: isEditable ? 'pointer' : 'default',
        transition: 'all 0.2s ease',
        border: '1px solid',
        borderColor: isExpanded
          ? 'rgba(74, 144, 217, 0.3)'
          : 'rgba(255, 255, 255, 0.1)',
        '&:hover': {
          backgroundColor: isEditable
            ? 'rgba(255, 255, 255, 0.08)'
            : 'rgba(255, 255, 255, 0.05)',
          borderColor: isEditable
            ? 'rgba(74, 144, 217, 0.2)'
            : 'rgba(255, 255, 255, 0.1)',
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
                  Click to expand and edit
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
            borderTop: '1px solid rgba(255, 255, 255, 0.1)',
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <Typography
            variant="subtitle2"
            sx={{
              color: 'text.secondary',
              mt: 2,
              mb: 1,
              fontSize: '0.8rem',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
            }}
          >
            Research Query Description
          </Typography>
          <TextField
            fullWidth
            multiline
            minRows={3}
            maxRows={6}
            value={editedDescription}
            onChange={handleDescriptionChange}
            placeholder="Edit the research query description..."
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
                lineHeight: 1.6,
              },
            }}
          />
        </Box>
      </Collapse>
    </Box>
  );
};

export default QueryCard;

/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * SingleChoiceWidget — MUI-styled single selection widget.
 * For simple choices: clickable button cards.
 * For rich choices (with descriptions): MUI Select dropdown + detail area.
 */

import React, { useState } from 'react';
import {
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Typography,
  TextField,
  ListItemText,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { Send as SendIcon } from '@mui/icons-material';
import { MarkdownRenderer } from '../common/MarkdownRenderer';

export default function SingleChoiceWidget({ config, onSubmit }) {
  const options = config?.input_mode?.options || config?.options || config?.choices || [];
  const allowCustom = config?.input_mode?.allow_custom ?? config?.allow_custom ?? true;
  const prompt = config?.input_mode?.prompt || config?.prompt || config?.title || '';

  // Detect if choices have descriptions (rich mode)
  const hasDescriptions = options.some(opt => opt.description);

  if (hasDescriptions) {
    return (
      <RichChoiceSelector
        options={options}
        prompt={prompt}
        allowCustom={allowCustom}
        onSubmit={onSubmit}
        config={config}
      />
    );
  }

  return (
    <SimpleChoiceSelector
      options={options}
      prompt={prompt}
      allowCustom={allowCustom}
      onSubmit={onSubmit}
    />
  );
}

/**
 * Rich choice selector — MUI Select dropdown with descriptions.
 * Matches the PlanModeSelector visual pattern from the demo experiment UI.
 */
function RichChoiceSelector({ options, prompt, allowCustom, onSubmit, config }) {
  const theme = useTheme();
  const [selectedIndex, setSelectedIndex] = useState(0);

  // Variable content from backend (for editable text block)
  const variableContent = config?.input_mode?.metadata?.variable_content || config?.metadata?.variable_content;
  const variableName = config?.input_mode?.metadata?.variable_name || config?.metadata?.variable_name;

  const getContent = (idx) =>
    variableContent?.[options[idx]?.value] || options[idx]?.description || '';

  const [editedContent, setEditedContent] = useState(() => getContent(0));

  const handleSubmit = () => {
    if (selectedIndex !== null && selectedIndex >= 0 && selectedIndex < options.length) {
      const payload = { choice_index: selectedIndex };
      const originalForSelection = getContent(selectedIndex);
      if (variableName && editedContent !== originalForSelection) {
        payload.variable_override = { [variableName]: editedContent };
      }
      onSubmit(payload);
    }
  };

  const selectedOption = options[selectedIndex] || null;

  return (
    <Box>
      {prompt && (
        <Box sx={{ mb: 1.5, '& p': { m: 0 } }}>
          <MarkdownRenderer content={prompt} />
        </Box>
      )}

      {/* Dropdown selector */}
      <FormControl fullWidth variant="outlined" sx={{ mb: 2 }}>
        <InputLabel
          sx={{
            color: 'text.secondary',
            '&.Mui-focused': { color: 'primary.main' },
          }}
        >
          Select
        </InputLabel>
        <Select
          value={selectedIndex}
          onChange={(e) => {
            setSelectedIndex(e.target.value);
            setEditedContent(getContent(e.target.value));
          }}
          label="Select"
          sx={{
            backgroundColor: theme.custom.surfaces.inputBg,
            '& .MuiOutlinedInput-notchedOutline': {
              borderColor: theme.custom.surfaces.inputBorder,
            },
            '&:hover .MuiOutlinedInput-notchedOutline': {
              borderColor: theme.custom.surfaces.inputBorderHover,
            },
            '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
              borderColor: 'primary.main',
            },
          }}
          renderValue={(idx) => {
            const opt = options[idx];
            return opt ? (
              <Box>
                <Typography variant="body1" sx={{ fontWeight: 600 }}>
                  {opt.label}
                </Typography>
                {opt.description && (
                  <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                    {opt.description}
                  </Typography>
                )}
              </Box>
            ) : '';
          }}
        >
          {options.map((opt, idx) => (
            <MenuItem
              key={idx}
              value={idx}
              sx={{
                py: 1.5,
                '&.Mui-selected': {
                  backgroundColor: theme.custom.surfaces.activeHighlight,
                },
              }}
            >
              <ListItemText
                primary={opt.label}
                secondary={opt.description || ''}
                primaryTypographyProps={{ fontWeight: 600, fontSize: '0.95rem' }}
                secondaryTypographyProps={{ fontSize: '0.8rem', color: 'text.secondary' }}
              />
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {/* Strategy details — editable if variable content available, read-only otherwise */}
      {(variableContent || selectedOption?.description) && (
        variableContent ? (
          <TextField
            fullWidth
            multiline
            minRows={6}
            maxRows={12}
            value={editedContent}
            onChange={(e) => setEditedContent(e.target.value)}
            label="Strategy Details"
            helperText="Optional — you can leave this as-is or customize"
            variant="outlined"
            sx={{
              mb: 2,
              '& .MuiOutlinedInput-root': {
                backgroundColor: theme.custom.surfaces.cardBg,
                fontFamily: "'Fira Code', 'Monaco', 'Consolas', monospace",
                fontSize: '0.85rem',
                lineHeight: 1.6,
                color: 'text.primary',
                '& fieldset': { borderColor: theme.custom.surfaces.overlayActive },
                '&:hover fieldset': { borderColor: theme.custom.surfaces.inputBorderHover },
                '&.Mui-focused fieldset': { borderColor: 'primary.main' },
              },
              '& .MuiInputLabel-root': { color: 'text.secondary' },
              '& .MuiFormHelperText-root': { color: 'text.disabled' },
            }}
          />
        ) : (
          <Box
            sx={{
              p: 2,
              mb: 2,
              backgroundColor: theme.custom.surfaces.cardBg,
              borderRadius: 1,
              border: '1px solid',
              borderColor: theme.custom.surfaces.overlayMedium,
              '& p': { m: 0 },
            }}
          >
            <Typography variant="caption" sx={{ color: 'text.disabled', mb: 0.5, display: 'block' }}>
              Strategy Details
            </Typography>
            <MarkdownRenderer content={selectedOption?.description || ''} />
          </Box>
        )
      )}

      <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          variant="contained"
          size="small"
          onClick={handleSubmit}
          disabled={selectedIndex === null}
          endIcon={<SendIcon sx={{ fontSize: 16 }} />}
          sx={{ textTransform: 'none', px: 2, py: 0.5, fontSize: '0.85rem' }}
        >
          Confirm Selection
        </Button>
      </Box>
    </Box>
  );
}

/**
 * Simple choice selector — MUI-styled clickable button cards.
 * Used for choices without descriptions.
 */
function SimpleChoiceSelector({ options, prompt, allowCustom, onSubmit }) {
  const theme = useTheme();
  const [selected, setSelected] = useState(null);
  const [customText, setCustomText] = useState('');
  const [showCustom, setShowCustom] = useState(false);

  const handleSubmit = () => {
    if (showCustom && customText.trim()) {
      onSubmit({ custom_text: customText.trim() });
    } else if (selected !== null) {
      onSubmit({ choice_index: selected });
    }
  };

  return (
    <Box>
      {prompt && (
        <Box sx={{ mb: 1.5, '& p': { m: 0 } }}>
          <MarkdownRenderer content={prompt} />
        </Box>
      )}

      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
        {options.map((opt, i) => (
          <Button
            key={i}
            variant={selected === i ? 'contained' : 'outlined'}
            onClick={() => { setSelected(i); setShowCustom(false); }}
            sx={{
              textTransform: 'none',
              py: 1,
              px: 2,
              fontSize: '0.85rem',
              borderColor: selected === i ? 'primary.main' : theme.custom.surfaces.inputBorder,
              color: selected === i ? 'white' : 'text.primary',
            }}
          >
            {opt.label}
          </Button>
        ))}
        {allowCustom && (
          <Button
            variant={showCustom ? 'contained' : 'outlined'}
            onClick={() => { setShowCustom(true); setSelected(null); }}
            sx={{
              textTransform: 'none',
              py: 1,
              px: 2,
              fontSize: '0.85rem',
              borderColor: showCustom ? 'primary.main' : theme.custom.surfaces.inputBorder,
              color: showCustom ? 'white' : 'text.secondary',
            }}
          >
            Custom...
          </Button>
        )}
      </Box>

      {showCustom && (
        <TextField
          fullWidth
          placeholder="Type your response..."
          value={customText}
          onChange={(e) => setCustomText(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
          autoFocus
          variant="outlined"
          size="small"
          sx={{
            mb: 2,
            '& .MuiOutlinedInput-root': {
              backgroundColor: theme.custom.surfaces.inputBg,
              '& fieldset': { borderColor: theme.custom.surfaces.inputBorder },
              '&.Mui-focused fieldset': { borderColor: 'primary.main' },
            },
            '& .MuiInputBase-input': { color: 'text.primary' },
          }}
        />
      )}

      <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          variant="contained"
          size="small"
          onClick={handleSubmit}
          disabled={selected === null && !(showCustom && customText.trim())}
          endIcon={<SendIcon sx={{ fontSize: 16 }} />}
          sx={{ textTransform: 'none', px: 2, py: 0.5, fontSize: '0.85rem' }}
        >
          Submit
        </Button>
      </Box>
    </Box>
  );
}

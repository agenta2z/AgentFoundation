/**
 * SingleChoiceWidget — single selection widget.
 * Adapted from AgentFoundation/ui/webui/react/src/components/widgets/SingleChoiceWidget.js
 *
 * Simple choices (no descriptions) → clickable button cards.
 * Rich choices (with descriptions) → MUI Select dropdown + detail area.
 *
 * Props:
 *   config.input_mode.prompt      - string
 *   config.input_mode.options     - [{label, value, description?}]
 *   config.input_mode.allow_custom - bool
 *   onSubmit({choice_index}) | onSubmit({custom_text})
 */

import React, { useState } from 'react';
import {
  Box, Button, FormControl, InputLabel, Select, MenuItem,
  TextField, Typography, ListItemText,
} from '@mui/material';
import { Send as SendIcon } from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import { MarkdownRenderer } from '../common/MarkdownRenderer';

export default function SingleChoiceWidget({ config, onSubmit }) {
  const options = config?.input_mode?.options || config?.options || [];
  const allowCustom = config?.input_mode?.allow_custom ?? true;
  const prompt = config?.input_mode?.prompt || config?.prompt || '';
  const hasDescriptions = options.some(opt => opt.description);

  if (hasDescriptions) {
    return <RichChoiceSelector options={options} prompt={prompt} allowCustom={allowCustom} onSubmit={onSubmit} config={config} />;
  }
  return <SimpleChoiceSelector options={options} prompt={prompt} allowCustom={allowCustom} onSubmit={onSubmit} />;
}

function RichChoiceSelector({ options, prompt, allowCustom, onSubmit, config }) {
  const theme = useTheme();
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [submitted, setSubmitted] = useState(false);
  const variableContent = config?.input_mode?.metadata?.variable_content;
  const variableName = config?.input_mode?.metadata?.variable_name;
  const getContent = (idx) => variableContent?.[options[idx]?.value] || options[idx]?.description || '';
  const [editedContent, setEditedContent] = useState(() => getContent(0));

  const handleSubmit = () => {
    if (submitted) return; // double-submit guard
    setSubmitted(true);
    const payload = { choice_index: selectedIndex };
    const orig = getContent(selectedIndex);
    if (variableName && editedContent !== orig) {
      payload.variable_override = { [variableName]: editedContent };
    }
    onSubmit(payload);
  };

  if (submitted) {
    return (
      <Box sx={{ color: 'primary.main', fontWeight: 500, fontSize: '0.9rem', py: 0.5 }}>
        Selected: {options[selectedIndex]?.label || 'Selection confirmed'}
      </Box>
    );
  }

  const selectedOption = options[selectedIndex] || null;

  return (
    <Box>
      {prompt && <Box sx={{ mb: 1.5, '& p': { m: 0 } }}><MarkdownRenderer content={prompt} /></Box>}

      <FormControl fullWidth variant="outlined" sx={{ mb: 2 }}>
        <InputLabel sx={{ color: 'text.secondary', '&.Mui-focused': { color: 'primary.main' } }}>Select</InputLabel>
        <Select
          value={selectedIndex}
          onChange={(e) => { setSelectedIndex(e.target.value); setEditedContent(getContent(e.target.value)); }}
          label="Select"
          sx={{
            backgroundColor: theme.custom?.surfaces?.inputBg || 'rgba(0,0,0,0.2)',
            '& .MuiOutlinedInput-notchedOutline': { borderColor: theme.custom?.surfaces?.inputBorder || 'rgba(255,255,255,0.2)' },
            '&.Mui-focused .MuiOutlinedInput-notchedOutline': { borderColor: 'primary.main' },
          }}
          renderValue={(idx) => {
            const opt = options[idx];
            return opt ? (
              <Box>
                <Typography variant="body1" sx={{ fontWeight: 600 }}>{opt.label}</Typography>
                {opt.description && <Typography variant="caption" sx={{ color: 'text.secondary' }}>{opt.description}</Typography>}
              </Box>
            ) : '';
          }}
        >
          {options.map((opt, idx) => (
            <MenuItem key={idx} value={idx} sx={{ py: 1.5, '&.Mui-selected': { backgroundColor: theme.custom?.surfaces?.activeHighlight || 'rgba(74,144,217,0.15)' } }}>
              <ListItemText primary={opt.label} secondary={opt.description || ''} primaryTypographyProps={{ fontWeight: 600 }} secondaryTypographyProps={{ fontSize: '0.8rem', color: 'text.secondary' }} />
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {(variableContent || selectedOption?.description) && (
        variableContent ? (
          <TextField
            fullWidth multiline minRows={4} maxRows={10}
            value={editedContent}
            onChange={(e) => setEditedContent(e.target.value)}
            label="Details" helperText="Optional — customize or leave as-is"
            variant="outlined"
            sx={{
              mb: 2,
              '& .MuiOutlinedInput-root': { backgroundColor: theme.custom?.surfaces?.cardBg || 'rgba(0,0,0,0.15)', fontSize: '0.85rem' },
              '& .MuiInputLabel-root': { color: 'text.secondary' },
              '& .MuiFormHelperText-root': { color: 'text.disabled' },
            }}
          />
        ) : (
          <Box sx={{ p: 1.5, mb: 2, backgroundColor: theme.custom?.surfaces?.cardBg || 'rgba(0,0,0,0.15)', borderRadius: 1, border: '1px solid', borderColor: theme.custom?.surfaces?.overlayMedium || 'rgba(255,255,255,0.1)', '& p': { m: 0 } }}>
            <Typography variant="caption" sx={{ color: 'text.disabled', mb: 0.5, display: 'block' }}>Details</Typography>
            <MarkdownRenderer content={selectedOption?.description || ''} />
          </Box>
        )
      )}

      <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
        <Button variant="contained" size="small" onClick={handleSubmit} disabled={selectedIndex === null} endIcon={<SendIcon sx={{ fontSize: 16 }} />} sx={{ textTransform: 'none', px: 2 }}>
          Confirm Selection
        </Button>
      </Box>
    </Box>
  );
}

function SimpleChoiceSelector({ options, prompt, allowCustom, onSubmit }) {
  const theme = useTheme();
  const [selected, setSelected] = useState(null);
  const [customText, setCustomText] = useState('');
  const [showCustom, setShowCustom] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = () => {
    if (submitted) return; // double-submit guard
    setSubmitted(true);
    if (showCustom && customText.trim()) {
      onSubmit({ custom_text: customText.trim() });
    } else if (selected !== null) {
      onSubmit({ choice_index: selected });
    }
  };

  if (submitted) {
    const label = selected !== null ? options[selected]?.label : customText.trim();
    return (
      <Box sx={{ color: 'primary.main', fontWeight: 500, fontSize: '0.9rem', py: 0.5 }}>
        Selected: {label || 'Custom response'}
      </Box>
    );
  }

  return (
    <Box>
      {prompt && <Box sx={{ mb: 1.5, '& p': { m: 0 } }}><MarkdownRenderer content={prompt} /></Box>}

      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 1.5 }}>
        {options.map((opt, i) => (
          <Button key={i} variant={selected === i ? 'contained' : 'outlined'}
            onClick={() => { setSelected(i); setShowCustom(false); }}
            sx={{ textTransform: 'none', py: 0.75, px: 2, fontSize: '0.85rem', borderColor: selected === i ? 'primary.main' : theme.custom?.surfaces?.inputBorder || 'rgba(255,255,255,0.2)' }}
          >
            {opt.label}
          </Button>
        ))}
        {allowCustom && (
          <Button variant={showCustom ? 'contained' : 'outlined'}
            onClick={() => { setShowCustom(true); setSelected(null); }}
            sx={{ textTransform: 'none', py: 0.75, px: 2, fontSize: '0.85rem', borderColor: showCustom ? 'primary.main' : theme.custom?.surfaces?.inputBorder || 'rgba(255,255,255,0.2)', color: showCustom ? 'white' : 'text.secondary' }}
          >
            Custom...
          </Button>
        )}
      </Box>

      {showCustom && (
        <TextField fullWidth placeholder="Type your response..." value={customText}
          onChange={(e) => setCustomText(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
          autoFocus variant="outlined" size="small"
          sx={{ mb: 1.5, '& .MuiOutlinedInput-root': { backgroundColor: theme.custom?.surfaces?.inputBg || 'rgba(0,0,0,0.2)', '& fieldset': { borderColor: theme.custom?.surfaces?.inputBorder || 'rgba(255,255,255,0.2)' }, '&.Mui-focused fieldset': { borderColor: 'primary.main' } }, '& .MuiInputBase-input': { color: 'text.primary' } }}
        />
      )}

      <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
        <Button variant="contained" size="small" onClick={handleSubmit}
          disabled={selected === null && !(showCustom && customText.trim())}
          endIcon={<SendIcon sx={{ fontSize: 16 }} />}
          sx={{ textTransform: 'none', px: 2 }}>
          Submit
        </Button>
      </Box>
    </Box>
  );
}

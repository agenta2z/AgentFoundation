/**
 * SingleChoiceWidget — single selection widget.
 * Adapted from AgentFoundation/ui/webui/react/src/components/widgets/SingleChoiceWidget.js
 *
 * Simple choices (no descriptions / inputs) → clickable button cards.
 * Rich choices (with descriptions or a nested `input` spec) → MUI Select dropdown + detail area.
 *
 * Composite choices: an option may carry a nested `input` spec
 *   { name, expected_input_type, prefix, allow_multiple_input, required, placeholder, serialization }.
 * The nested input is revealed only when that option is selected; on submit the widget emits
 *   { choice_index, inputs: { [option.input.name]: <value> } }
 * where <value> is a string (single) or array of strings (multi). Plain options still emit
 * { choice_index } with no inputs. See conditional_path_inputs_plan.md Scenario B.
 *
 * Props:
 *   config.input_mode.prompt              - string
 *   config.input_mode.options             - [{label, value, description?, input?}]
 *   config.input_mode.allow_custom        - bool
 *   config.pathAutocompleteProvider       - host-injected path provider (threaded down)
 *   onSubmit({choice_index}) | onSubmit({choice_index, inputs}) | onSubmit({custom_text})
 */

import React, { useState } from 'react';
import {
  Box, Button, FormControl, InputLabel, Select, MenuItem,
  TextField, Typography, ListItemText,
} from '@mui/material';
import { Send as SendIcon } from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import { MarkdownRenderer } from '../common/MarkdownRenderer';
import EllipsisText from '../common/EllipsisText';
import PathAutocompleteInput from './PathAutocompleteInput';
import MultiValueInput from './MultiValueInput';

/**
 * Renders the nested input for a composite choice option. Returns null for plain options.
 * Currently supports path inputs; falls through to a plain text field for other types.
 */
function NestedOptionInput({ spec, value, onChange, provider, disabled }) {
  if (!spec) return null;
  const placeholder = spec.placeholder || (spec.expected_input_type === 'path' ? 'relative/path' : 'Enter a value...');

  if (spec.allow_multiple_input) {
    const values = Array.isArray(value) ? value : [];
    return (
      <MultiValueInput
        values={values}
        onChange={onChange}
        addLabel="Add"
        disabled={disabled}
        renderInput={({ value: draft, onChange: setDraft }) => (
          <PathAutocompleteInput
            value={draft}
            onChange={setDraft}
            prefix={spec.prefix || ''}
            provider={provider}
            placeholder={placeholder}
            disabled={disabled}
          />
        )}
      />
    );
  }
  return (
    <PathAutocompleteInput
      value={typeof value === 'string' ? value : ''}
      onChange={onChange}
      prefix={spec.prefix || ''}
      provider={provider}
      placeholder={placeholder}
      disabled={disabled}
    />
  );
}

export default function SingleChoiceWidget({ config, onSubmit, submitLabel, readOnly, value }) {
  const options = config?.input_mode?.options || config?.options || [];
  const allowCustom = config?.input_mode?.allow_custom ?? true;
  const prompt = config?.input_mode?.prompt || config?.prompt || '';
  const hasDescriptions = options.some(opt => opt.description);
  const hasInputs = options.some(opt => opt.input);

  if (hasDescriptions || hasInputs) {
    return <RichChoiceSelector options={options} prompt={prompt} allowCustom={allowCustom} onSubmit={onSubmit} config={config} submitLabel={submitLabel} readOnly={readOnly} value={value} />;
  }
  return <SimpleChoiceSelector options={options} prompt={prompt} allowCustom={allowCustom} onSubmit={onSubmit} submitLabel={submitLabel} readOnly={readOnly} value={value} />;
}

function RichChoiceSelector({ options, prompt, allowCustom, onSubmit, config, submitLabel, readOnly, value }) {
  const theme = useTheme();
  // In read-only mode, seed all state from the committed value so the disabled
  // widget shows the user's selection (and any nested input) exactly as submitted.
  const initIndex = readOnly && value && value.choice_index != null ? value.choice_index : 0;
  const [selectedIndex, setSelectedIndex] = useState(initIndex);
  const [submitted, setSubmitted] = useState(false);
  // Whether the selected option's inline description is truncated inside the Select.
  // The read-only "Details" box is shown ONLY when it is — otherwise it would just
  // duplicate the fully-visible inline text.
  const [descTruncated, setDescTruncated] = useState(false);
  const variableContent = config?.input_mode?.metadata?.variable_content;
  const variableName = config?.input_mode?.metadata?.variable_name;
  const provider = config?.pathAutocompleteProvider;
  const getContent = (idx) => variableContent?.[options[idx]?.value] || options[idx]?.description || '';
  const [editedContent, setEditedContent] = useState(() => {
    if (readOnly && value && value.variable_override && typeof value.variable_override === 'object') {
      const vals = Object.values(value.variable_override);
      if (vals.length) return vals[0];
    }
    return getContent(initIndex);
  });
  // Per-option nested input values, keyed by option index. Only the selected
  // option's value is ever submitted, so stale values for other options are ignored.
  const [optionInputs, setOptionInputs] = useState(() => {
    if (readOnly && value && value.inputs && typeof value.inputs === 'object') {
      const vals = Object.values(value.inputs);
      return vals.length ? { [initIndex]: vals[0] } : {};
    }
    return {};
  });

  const selectedSpec = options[selectedIndex]?.input || null;

  const handleSubmit = () => {
    if (submitted) return; // double-submit guard
    setSubmitted(true);
    const payload = { choice_index: selectedIndex };
    const orig = getContent(selectedIndex);
    if (variableName && editedContent !== orig) {
      payload.variable_override = { [variableName]: editedContent };
    }
    // Composite choice: bind ONLY the selected option's nested input.
    if (selectedSpec?.name) {
      const raw = optionInputs[selectedIndex];
      const value = selectedSpec.allow_multiple_input ? (Array.isArray(raw) ? raw : []) : (raw || '');
      payload.inputs = { [selectedSpec.name]: value };
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

  // A required nested input must be non-empty before submission is allowed.
  const inputSatisfied = (() => {
    if (!selectedSpec?.required) return true;
    const raw = optionInputs[selectedIndex];
    return selectedSpec.allow_multiple_input ? (Array.isArray(raw) && raw.length > 0) : !!(raw && String(raw).trim());
  })();

  return (
    <Box>
      {prompt && <Box sx={{ mb: 1.5, '& p': { m: 0 } }}><MarkdownRenderer content={prompt} /></Box>}

      <FormControl fullWidth variant="outlined" sx={{ mb: 2 }}>
        <InputLabel sx={{ color: 'text.secondary', '&.Mui-focused': { color: 'primary.main' } }}>Select</InputLabel>
        <Select
          value={selectedIndex}
          onChange={(e) => { setSelectedIndex(e.target.value); setEditedContent(getContent(e.target.value)); }}
          label="Select"
          disabled={readOnly}
          sx={{
            backgroundColor: theme.custom?.surfaces?.inputBg || 'rgba(0,0,0,0.2)',
            '& .MuiOutlinedInput-notchedOutline': { borderColor: theme.custom?.surfaces?.inputBorder || 'rgba(255,255,255,0.2)' },
            '&.Mui-focused .MuiOutlinedInput-notchedOutline': { borderColor: 'primary.main' },
          }}
          renderValue={(idx) => {
            const opt = options[idx];
            return opt ? (
              <Box sx={{ minWidth: 0, overflow: 'hidden' }}>
                <Typography variant="body1" sx={{ fontWeight: 600 }}>{opt.label}</Typography>
                {opt.description && (
                  <EllipsisText
                    text={opt.description}
                    onTruncatedChange={setDescTruncated}
                    sx={{ color: 'text.secondary' }}
                  />
                )}
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

      {variableContent ? (
        <TextField
          fullWidth multiline minRows={4} maxRows={10}
          value={editedContent}
          onChange={(e) => setEditedContent(e.target.value)}
          disabled={readOnly}
          label="Details" helperText="Optional — customize or leave as-is"
          variant="outlined"
          sx={{
            mb: 2,
            '& .MuiOutlinedInput-root': { backgroundColor: theme.custom?.surfaces?.cardBg || 'rgba(0,0,0,0.15)', fontSize: '0.85rem' },
            '& .MuiInputLabel-root': { color: 'text.secondary' },
            '& .MuiFormHelperText-root': { color: 'text.disabled' },
          }}
        />
      ) : (selectedOption?.description && descTruncated) ? (
        // Shown only when the inline description was truncated in the Select above,
        // so this is the full text rather than a duplicate of what's already visible.
        <Box sx={{ p: 1.5, mb: 2, backgroundColor: theme.custom?.surfaces?.cardBg || 'rgba(0,0,0,0.15)', borderRadius: 1, border: '1px solid', borderColor: theme.custom?.surfaces?.overlayMedium || 'rgba(255,255,255,0.1)', '& p': { m: 0 } }}>
          <Typography variant="caption" sx={{ color: 'text.disabled', mb: 0.5, display: 'block' }}>Details</Typography>
          <MarkdownRenderer content={selectedOption?.description || ''} />
        </Box>
      ) : null}

      {/* Nested input — revealed only for the selected option that declares one. */}
      {selectedSpec && (
        <Box sx={{ mb: 2 }}>
          {(selectedSpec.label || selectedSpec.name) && (
            <Typography variant="caption" sx={{ color: 'text.secondary', mb: 0.5, display: 'block' }}>
              {selectedSpec.label || selectedSpec.name}{selectedSpec.required ? ' *' : ''}
            </Typography>
          )}
          <NestedOptionInput
            spec={selectedSpec}
            value={optionInputs[selectedIndex]}
            onChange={(val) => setOptionInputs(prev => ({ ...prev, [selectedIndex]: val }))}
            provider={provider}
            disabled={readOnly}
          />
        </Box>
      )}

      {!readOnly && (
        <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
          <Button variant="contained" size="small" onClick={handleSubmit} disabled={selectedIndex === null || !inputSatisfied} endIcon={<SendIcon sx={{ fontSize: 16 }} />} sx={{ textTransform: 'none', px: 2 }}>
            {submitLabel || 'Confirm Selection'}
          </Button>
        </Box>
      )}
    </Box>
  );
}

function SimpleChoiceSelector({ options, prompt, allowCustom, onSubmit, submitLabel, readOnly, value }) {
  const theme = useTheme();
  // Read-only mode seeds the selection (or custom text) from the committed value.
  const [selected, setSelected] = useState(
    readOnly && value && value.choice_index != null ? value.choice_index : null,
  );
  const [customText, setCustomText] = useState(
    readOnly && value && value.custom_text ? value.custom_text : '',
  );
  const [showCustom, setShowCustom] = useState(!!(readOnly && value && value.custom_text));
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
            disabled={readOnly}
            onClick={() => { setSelected(i); setShowCustom(false); }}
            sx={{ textTransform: 'none', py: 0.75, px: 2, fontSize: '0.85rem', borderColor: selected === i ? 'primary.main' : theme.custom?.surfaces?.inputBorder || 'rgba(255,255,255,0.2)' }}
          >
            {opt.label}
          </Button>
        ))}
        {!readOnly && allowCustom && (
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
          autoFocus={!readOnly} disabled={readOnly} variant="outlined" size="small"
          sx={{ mb: 1.5, '& .MuiOutlinedInput-root': { backgroundColor: theme.custom?.surfaces?.inputBg || 'rgba(0,0,0,0.2)', '& fieldset': { borderColor: theme.custom?.surfaces?.inputBorder || 'rgba(255,255,255,0.2)' }, '&.Mui-focused fieldset': { borderColor: 'primary.main' } }, '& .MuiInputBase-input': { color: 'text.primary' } }}
        />
      )}

      {!readOnly && (
        <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
          <Button variant="contained" size="small" onClick={handleSubmit}
            disabled={selected === null && !(showCustom && customText.trim())}
            endIcon={<SendIcon sx={{ fontSize: 16 }} />}
            sx={{ textTransform: 'none', px: 2 }}>
            {submitLabel || 'Submit'}
          </Button>
        </Box>
      )}
    </Box>
  );
}

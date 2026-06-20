/**
 * PathInputWidget — registered widget for `metadata.widget_type === "path_input"`.
 *
 * Renders a path autocomplete for direct path clarifications. Reads the typed fields the
 * backend now sets on the input mode:
 *   config.input_mode.expected_input_type  ("free_text" | "path")
 *   config.input_mode.prefix               (concrete absolute path)
 *   config.input_mode.allow_multiple_input (bool)
 * and the host-injected provider (threaded as config.pathAutocompleteProvider).
 *
 * onSubmit shapes (see conditional_path_inputs_plan.md, Scenario A):
 *   single value   → { content: "<relative path>" }
 *   multiple values → { content: ["p1", "p2"] }
 *
 * Degrades to a plain text field when no provider is available (PathAutocompleteInput
 * handles that internally).
 */

import React, { useState } from 'react';
import { Box, Button, Typography } from '@mui/material';
import { Send as SendIcon } from '@mui/icons-material';
import { MarkdownRenderer } from '../common/MarkdownRenderer';
import PathAutocompleteInput from './PathAutocompleteInput';
import MultiValueInput from './MultiValueInput';

export default function PathInputWidget({ config, onSubmit }) {
  const inputMode = config?.input_mode || {};
  const prompt = inputMode.prompt || config?.prompt || '';
  const prefix = inputMode.prefix || inputMode.metadata?.prefix || '';
  const allowMultiple = !!inputMode.allow_multiple_input;
  const placeholder = config?.placeholder || inputMode.placeholder || 'relative/path';
  const provider = config?.pathAutocompleteProvider;

  const [single, setSingle] = useState('');
  const [values, setValues] = useState([]);
  const [submitted, setSubmitted] = useState(false);

  const canSubmit = allowMultiple ? values.length > 0 : !!single.trim();

  const handleSubmit = () => {
    if (submitted || !canSubmit) return; // double-submit guard
    setSubmitted(true);
    onSubmit(allowMultiple ? { content: values } : { content: single.trim() });
  };

  if (submitted) {
    const shown = allowMultiple ? values.join(', ') : single.trim();
    return (
      <Box sx={{ color: 'text.secondary', fontStyle: 'italic', fontSize: '0.9rem', py: 0.5 }}>
        "{shown}"
      </Box>
    );
  }

  return (
    <Box>
      {prompt && (
        <Box sx={{ mb: 1.5, '& p': { m: 0 } }}>
          <MarkdownRenderer content={prompt} />
        </Box>
      )}

      {allowMultiple ? (
        <MultiValueInput
          values={values}
          onChange={setValues}
          addLabel="Add path"
          renderInput={({ value, onChange }) => (
            <PathAutocompleteInput
              value={value}
              onChange={onChange}
              prefix={prefix}
              provider={provider}
              placeholder={placeholder}
            />
          )}
        />
      ) : (
        <PathAutocompleteInput
          value={single}
          onChange={setSingle}
          prefix={prefix}
          provider={provider}
          placeholder={placeholder}
          autoFocus
        />
      )}

      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 1 }}>
        <Typography variant="caption" sx={{ color: 'text.disabled' }}>
          {allowMultiple ? 'Add one or more paths' : 'Enter a path relative to the prefix'}
        </Typography>
        <Button
          variant="contained"
          size="small"
          onClick={handleSubmit}
          disabled={!canSubmit}
          endIcon={<SendIcon sx={{ fontSize: 16 }} />}
          sx={{ textTransform: 'none', px: 2, fontSize: '0.85rem' }}
        >
          Submit
        </Button>
      </Box>
    </Box>
  );
}

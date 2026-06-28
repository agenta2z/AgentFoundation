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

export default function PathInputWidget({ config, onSubmit, submitLabel, readOnly, value }) {
  const inputMode = config?.input_mode || {};
  const prompt = inputMode.prompt || config?.prompt || '';
  const prefix = inputMode.prefix || inputMode.metadata?.prefix || '';
  const allowMultiple = !!inputMode.allow_multiple_input;
  const placeholder = config?.placeholder || inputMode.placeholder || 'relative/path';
  const provider = config?.pathAutocompleteProvider;

  // Prefill: a default the agent supplied (e.g. a path the user already gave in
  // their request). Accept absolute OR relative — strip the prefix so the box
  // shows the relative remainder (the prefix renders as its own adornment).
  const rawDefault = inputMode.default ?? inputMode.metadata?.default;
  const stripPrefix = (s) => {
    if (typeof s !== 'string') return '';
    return prefix && s.startsWith(prefix) ? s.slice(prefix.length).replace(/^\/+/, '') : s;
  };
  const defaultSingle = !allowMultiple ? stripPrefix(rawDefault) : '';
  const defaultValues = allowMultiple
    ? (Array.isArray(rawDefault) ? rawDefault.map(stripPrefix) : (rawDefault ? [stripPrefix(rawDefault)] : []))
    : [];

  // Read-only mode shows the committed value(s) disabled (no submit row);
  // otherwise seed any prefill so the user can just confirm.
  const [single, setSingle] = useState(
    readOnly && value && typeof value.content === 'string' ? value.content : defaultSingle,
  );
  const [values, setValues] = useState(
    readOnly && value && Array.isArray(value.content) ? value.content : defaultValues,
  );
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
          disabled={readOnly}
          renderInput={({ value: draft, onChange }) => (
            <PathAutocompleteInput
              value={draft}
              onChange={onChange}
              prefix={prefix}
              provider={provider}
              placeholder={placeholder}
              disabled={readOnly}
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
          autoFocus={!readOnly}
          disabled={readOnly}
        />
      )}

      {!readOnly && (
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
            {submitLabel || 'Submit'}
          </Button>
        </Box>
      )}
    </Box>
  );
}

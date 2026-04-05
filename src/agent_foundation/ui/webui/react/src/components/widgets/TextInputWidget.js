/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * TextInputWidget — MUI-styled text input with markdown prompt.
 * Supports two modes:
 *   - free_text: Plain multiline text input (default)
 *   - path: Path input with prefix display and filesystem autocomplete
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Box,
  TextField,
  Button,
  Typography,
  Chip,
  Autocomplete,
  InputAdornment,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import {
  Send as SendIcon,
  Folder as FolderIcon,
} from '@mui/icons-material';
import { MarkdownRenderer } from '../common/MarkdownRenderer';

/**
 * Debounce hook — delays calling fn until after wait ms of inactivity.
 */
function useDebounce(value, delay) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);
  return debounced;
}

/**
 * Path input variant with prefix display and filesystem autocomplete.
 */
function PathInput({ prefix, prompt, onSubmit }) {
  const theme = useTheme();
  const [inputValue, setInputValue] = useState('');
  const [options, setOptions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [validity, setValidity] = useState(null); // null=unchecked, true=valid, false=invalid
  const debouncedInput = useDebounce(inputValue, 300);
  const abortRef = useRef(null);

  // Fetch suggestions when debounced input changes
  useEffect(() => {
    if (!prefix) return;
    if (!debouncedInput.trim()) {
      setOptions([]);
      setValidity(null);
      return;
    }

    // Abort previous request
    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    const params = new URLSearchParams({ prefix, partial: debouncedInput, dirs_only: 'false' });
    fetch(`/api/workspace/path-complete?${params}`, { signal: controller.signal })
      .then(r => r.json())
      .then(data => {
        const suggestions = data.suggestions || [];
        setOptions(suggestions);
        setLoading(false);

        // Validate: check if the typed path IS a valid directory
        // A path is valid if it ends with '/' and matches an exact suggestion,
        // or if the full path (prefix + input) was the search dir (suggestions came back)
        const normalizedInput = debouncedInput.replace(/\/$/, '');
        const exactMatch = suggestions.some(
          s => s.path.replace(/\/$/, '') === normalizedInput
        );
        // Also valid if the input IS a directory (suggestions came back for it as a parent)
        const isParentDir = debouncedInput.endsWith('/') && suggestions.length > 0;

        if (exactMatch || isParentDir) {
          setValidity(true);
        } else if (suggestions.length === 0 && debouncedInput.length > 2) {
          setValidity(false);
        } else {
          setValidity(null); // Still typing, partial matches exist
        }
      })
      .catch(err => {
        if (err.name !== 'AbortError') {
          setLoading(false);
          setValidity(false);
        }
      });
  }, [debouncedInput, prefix]);

  const handleSubmit = () => {
    const fullPath = prefix
      ? `${prefix.replace(/\/$/, '')}/${inputValue.replace(/^\//, '')}`
      : inputValue;
    if (fullPath.trim()) {
      onSubmit({ content: fullPath.trim() });
    }
  };

  return (
    <Box>
      {prompt && (
        <Box sx={{ mb: 1.5, '& p': { m: 0 } }}>
          <MarkdownRenderer content={prompt} />
        </Box>
      )}

      {/* Prefix display */}
      {prefix && (
        <Chip
          icon={<FolderIcon sx={{ fontSize: 14 }} />}
          label={prefix.endsWith('/') ? prefix : prefix + '/'}
          size="small"
          sx={{
            mb: 1,
            backgroundColor: theme.custom.surfaces.overlayMedium,
            color: 'text.secondary',
            fontFamily: "'Fira Code', 'Monaco', monospace",
            fontSize: '0.8rem',
            height: 24,
          }}
        />
      )}

      <Autocomplete
        freeSolo
        options={options.map(o => o.path)}
        inputValue={inputValue}
        onInputChange={(_, newValue) => setInputValue(newValue)}
        onChange={(_, newValue) => {
          if (newValue) setInputValue(newValue);
        }}
        loading={loading}
        filterOptions={(x) => x}  // Server-side filtering
        renderOption={(props, option) => {
          const { key, ...rest } = props;
          const isDir = option.endsWith('/');
          return (
            <li key={key} {...rest} style={{ fontFamily: "'Fira Code', monospace", fontSize: '0.85rem' }}>
              {isDir
                ? <FolderIcon sx={{ fontSize: 14, mr: 1, color: 'warning.main' }} />
                : <Box component="span" sx={{ fontSize: 14, mr: 1, color: 'text.disabled' }}>📄</Box>
              }
              {option}
            </li>
          );
        }}
        renderInput={(params) => (
          <TextField
            {...params}
            placeholder="ads/ranking/models/..."
            autoFocus
            variant="outlined"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey && inputValue.trim()) {
                e.preventDefault();
                handleSubmit();
              }
            }}
            InputProps={{
              ...params.InputProps,
              startAdornment: prefix ? (
                <InputAdornment position="start">
                  <Typography
                    variant="body2"
                    sx={{
                      color: 'text.disabled',
                      fontFamily: "'Fira Code', monospace",
                      fontSize: '0.85rem',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    .../
                  </Typography>
                </InputAdornment>
              ) : undefined,
            }}
            sx={{
              '& .MuiOutlinedInput-root': {
                backgroundColor: theme.custom.surfaces.inputBg,
                fontFamily: "'Fira Code', 'Monaco', monospace",
                fontSize: '0.9rem',
                '& fieldset': {
                  borderColor: validity === false
                    ? 'error.main'
                    : validity === true
                      ? 'success.main'
                      : theme.custom.surfaces.inputBorder,
                },
                '&:hover fieldset': {
                  borderColor: validity === false
                    ? 'error.light'
                    : validity === true
                      ? 'success.light'
                      : theme.custom.surfaces.inputBorderHover,
                },
                '&.Mui-focused fieldset': {
                  borderColor: validity === false
                    ? 'error.main'
                    : validity === true
                      ? 'success.main'
                      : 'primary.main',
                },
              },
              '& .MuiInputBase-input': { color: 'text.primary' },
            }}
          />
        )}
        sx={{
          '& .MuiAutocomplete-listbox': {
            backgroundColor: 'background.paper',
            maxHeight: 200,
          },
          '& .MuiAutocomplete-option': {
            '&:hover': { backgroundColor: theme.custom.surfaces.overlayMedium },
          },
        }}
      />

      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 1 }}>
        <Typography variant="caption" sx={{ color: validity === false ? 'error.main' : 'text.disabled' }}>
          {validity === false
            ? 'Directory not found'
            : validity === true
              ? 'Valid directory'
              : 'Type to search directories. Press Enter to submit.'}
        </Typography>
        <Button
          variant="contained"
          size="small"
          onClick={handleSubmit}
          disabled={!inputValue.trim()}
          endIcon={<SendIcon sx={{ fontSize: 16 }} />}
          sx={{ textTransform: 'none', px: 2, py: 0.5, fontSize: '0.85rem' }}
        >
          Set Path
        </Button>
      </Box>
    </Box>
  );
}

/**
 * Main TextInputWidget — dispatches to PathInput or plain text input.
 */
export default function TextInputWidget({ config, onSubmit }) {
  const metadata = config?.input_mode?.metadata || config?.metadata || {};
  const expectedType = metadata.expected_input_type || config?.expected_input_type || 'free_text';
  const prefix = metadata.prefix || config?.prefix || '';
  const prompt = config?.input_mode?.prompt || config?.prompt || config?.title || '';

  // Path mode
  if (expectedType === 'path') {
    return <PathInput prefix={prefix} prompt={prompt} onSubmit={onSubmit} />;
  }

  // Default: free text mode
  return <FreeTextInput prompt={prompt} config={config} onSubmit={onSubmit} />;
}

function FreeTextInput({ prompt, config, onSubmit }) {
  const theme = useTheme();
  const [text, setText] = useState('');
  const placeholder = config?.placeholder || 'Type your response...';

  const handleSubmit = () => {
    if (text.trim()) {
      onSubmit({ content: text.trim() });
    }
  };

  return (
    <Box>
      {prompt && (
        <Box sx={{ mb: 1.5, '& p': { m: 0 } }}>
          <MarkdownRenderer content={prompt} />
        </Box>
      )}
      <TextField
        fullWidth
        multiline
        minRows={2}
        maxRows={6}
        placeholder={placeholder}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
          }
        }}
        autoFocus
        variant="outlined"
        sx={{
          '& .MuiOutlinedInput-root': {
            backgroundColor: theme.custom.surfaces.inputBg,
            '& fieldset': { borderColor: theme.custom.surfaces.inputBorder },
            '&:hover fieldset': { borderColor: theme.custom.surfaces.inputBorderHover },
            '&.Mui-focused fieldset': { borderColor: 'primary.main' },
          },
          '& .MuiInputBase-input': {
            color: 'text.primary',
            fontSize: '0.95rem',
          },
        }}
      />
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 1 }}>
        <Button
          variant="contained"
          size="small"
          onClick={handleSubmit}
          disabled={!text.trim()}
          endIcon={<SendIcon sx={{ fontSize: 16 }} />}
          sx={{ textTransform: 'none', px: 2, py: 0.5, fontSize: '0.85rem' }}
        >
          Submit
        </Button>
      </Box>
      <Typography variant="caption" sx={{ color: 'text.disabled', mt: 0.5, display: 'block' }}>
        Press Enter to submit, Shift+Enter for new line
      </Typography>
    </Box>
  );
}

/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * ToolConfigPanel — renders tool parameters as form elements.
 * Driven by tool.json parameter definitions passed via config.
 * Used inside ConfirmationWidget for inline parameter configuration.
 */

import React from 'react';
import {
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  TextField,
  Typography,
  Tabs,
  Tab,
  Divider,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';

/**
 * Render a single parameter as the appropriate form element.
 */
function ParamField({ param, value, onChange }) {
  const theme = useTheme();
  const name = param.name.replace(/^--/, '');
  const label = name.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

  // Flag → Switch toggle
  if (param.type === 'flag') {
    return (
      <FormControlLabel
        control={
          <Switch
            checked={!!value}
            onChange={(e) => onChange(name, e.target.checked)}
            size="small"
          />
        }
        label={
          <Box>
            <Typography variant="body2" sx={{ fontWeight: 500 }}>{label}</Typography>
            {param.description && (
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                {param.description}
              </Typography>
            )}
          </Box>
        }
        sx={{ alignItems: 'flex-start', mb: 1.5 }}
      />
    );
  }

  // String with choices → Select dropdown
  if (param.choices && param.choices.length > 0) {
    return (
      <FormControl fullWidth variant="outlined" size="small" sx={{ mb: 1.5 }}>
        <InputLabel sx={{ color: 'text.secondary', '&.Mui-focused': { color: 'primary.main' } }}>
          {label}
        </InputLabel>
        <Select
          value={value || ''}
          onChange={(e) => onChange(name, e.target.value)}
          label={label}
          sx={{
            backgroundColor: theme.custom.surfaces.inputBg,
            '& .MuiOutlinedInput-notchedOutline': { borderColor: theme.custom.surfaces.inputBorder },
            '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: theme.custom.surfaces.inputBorderHover },
            '&.Mui-focused .MuiOutlinedInput-notchedOutline': { borderColor: 'primary.main' },
          }}
        >
          <MenuItem value=""><em>Default</em></MenuItem>
          {param.choices.map((choice) => (
            <MenuItem key={choice} value={choice}>{choice}</MenuItem>
          ))}
        </Select>
        {param.description && (
          <Typography variant="caption" sx={{ color: 'text.disabled', mt: 0.5, display: 'block' }}>
            {param.description}
          </Typography>
        )}
      </FormControl>
    );
  }

  // String/path → TextField
  return (
    <TextField
      fullWidth
      size="small"
      label={label}
      placeholder={param.default != null ? String(param.default) : ''}
      value={value || ''}
      onChange={(e) => onChange(name, e.target.value)}
      variant="outlined"
      sx={{
        mb: 1.5,
        '& .MuiOutlinedInput-root': {
          backgroundColor: theme.custom.surfaces.inputBg,
          '& fieldset': { borderColor: theme.custom.surfaces.inputBorder },
          '&:hover fieldset': { borderColor: theme.custom.surfaces.inputBorderHover },
          '&.Mui-focused fieldset': { borderColor: 'primary.main' },
        },
        '& .MuiInputBase-input': { color: 'text.primary', fontSize: '0.85rem' },
        '& .MuiInputLabel-root': { color: 'text.secondary' },
      }}
      helperText={param.description}
      FormHelperTextProps={{ sx: { color: 'text.disabled' } }}
    />
  );
}

/**
 * ToolConfigPanel — tabbed parameter configuration.
 * Params with popular=true go in "Common" tab, others in "Advanced".
 */
export default function ToolConfigPanel({ params, values, onChange }) {
  const theme = useTheme();
  const [tabIndex, setTabIndex] = React.useState(0);

  if (!params || params.length === 0) return null;

  const commonParams = params.filter(p => p.popular && !p.positional);
  const advancedParams = params.filter(p => !p.popular && !p.positional);
  const hasTabs = commonParams.length > 0 && advancedParams.length > 0;

  const handleChange = (name, value) => {
    onChange({ ...values, [name]: value });
  };

  const renderParams = (paramList) =>
    paramList.map((param) => {
      const name = param.name.replace(/^--/, '');
      return (
        <ParamField
          key={name}
          param={param}
          value={values[name]}
          onChange={handleChange}
        />
      );
    });

  if (!hasTabs) {
    // No tab split needed — show all params in one list
    return (
      <Box sx={{ mt: 1.5 }}>
        <Divider sx={{ mb: 1.5, borderColor: theme.custom.surfaces.overlayActive }} />
        <Typography variant="caption" sx={{ color: 'text.secondary', mb: 1, display: 'block' }}>
          Configuration
        </Typography>
        {renderParams([...commonParams, ...advancedParams])}
      </Box>
    );
  }

  return (
    <Box sx={{ mt: 1.5 }}>
      <Divider sx={{ mb: 0.5, borderColor: theme.custom.surfaces.overlayActive }} />
      <Tabs
        value={tabIndex}
        onChange={(_, v) => setTabIndex(v)}
        sx={{
          minHeight: 32,
          '& .MuiTab-root': {
            minHeight: 32,
            py: 0.5,
            textTransform: 'none',
            fontSize: '0.8rem',
          },
        }}
      >
        <Tab label={`Common (${commonParams.length})`} />
        <Tab label={`Advanced (${advancedParams.length})`} />
      </Tabs>
      <Box sx={{ pt: 1.5, maxHeight: 300, overflowY: 'auto' }}>
        {tabIndex === 0 ? renderParams(commonParams) : renderParams(advancedParams)}
      </Box>
    </Box>
  );
}

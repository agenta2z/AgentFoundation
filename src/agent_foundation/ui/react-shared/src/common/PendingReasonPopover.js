/**
 * PendingReasonPopover — interactive resolution for pending tasks/employees.
 * Uses theme palette tokens instead of hardcoded colors.
 */

import React, { useState } from 'react';
import Popover from '@mui/material/Popover';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Radio from '@mui/material/Radio';
import RadioGroup from '@mui/material/RadioGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import TextField from '@mui/material/TextField';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import Divider from '@mui/material/Divider';
import Alert from '@mui/material/Alert';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import { useTheme, alpha } from '@mui/material/styles';

function relativeTime(isoString) {
  if (!isoString) return '';
  const now = new Date();
  const then = new Date(isoString);
  const diffMs = now - then;
  const diffHr = Math.floor(diffMs / 3600000);
  if (diffHr < 1) return 'just now';
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDays = Math.floor(diffHr / 24);
  if (diffDays === 1) return '1 day ago';
  return `${diffDays} days ago`;
}

function ResolutionOptions({ resolution, onResolve }) {
  const theme = useTheme();
  const [selectedOption, setSelectedOption] = useState('');
  const [inputValue, setInputValue] = useState('');
  const [resolved, setResolved] = useState(false);

  if (!resolution?.options?.length) return null;

  const selected = resolution.options.find(o => o.id === selectedOption);
  const canApply = selectedOption && (!selected?.requires_input || inputValue.trim().length > 0);

  const handleApply = () => { setResolved(true); if (onResolve) onResolve({ option: selectedOption, input: inputValue }); };

  if (resolved) {
    return (
      <Alert icon={<CheckCircleIcon sx={{ fontSize: 18 }} />} severity="success" sx={{
        mt: 1, backgroundColor: alpha(theme.palette.success.main, 0.08), color: theme.palette.success.light,
        border: `1px solid ${alpha(theme.palette.success.main, 0.2)}`, '& .MuiAlert-icon': { color: theme.palette.success.light },
      }}>
        Resolution applied: {selected?.label || selectedOption}
      </Alert>
    );
  }

  return (
    <Box sx={{ mt: 1.5 }}>
      <Divider sx={{ mb: 1.5 }} />
      <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary', textTransform: 'uppercase', letterSpacing: 0.3, fontSize: '0.6rem', display: 'block', mb: 1 }}>
        {resolution.prompt || 'How would you like to resolve this?'}
      </Typography>
      <RadioGroup value={selectedOption} onChange={(e) => { setSelectedOption(e.target.value); setInputValue(''); }}>
        {resolution.options.map((opt) => (
          <Box key={opt.id} sx={{ mb: 0.5 }}>
            <FormControlLabel value={opt.id}
              control={<Radio size="small" sx={{ py: 0.5, color: 'text.secondary', '&.Mui-checked': { color: theme.palette.secondary.main } }} />}
              label={<Box><Typography variant="body2" sx={{ fontSize: '0.8rem', fontWeight: 500, lineHeight: 1.3 }}>{opt.label}</Typography><Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.65rem', lineHeight: 1.3 }}>{opt.description}</Typography></Box>}
              sx={{ alignItems: 'flex-start', ml: 0, mr: 0 }}
            />
            {selectedOption === opt.id && opt.requires_input && (
              <Box sx={{ pl: 3.5, pb: 1 }}>
                {opt.input_type === 'select' ? (
                  <Select value={inputValue} onChange={(e) => setInputValue(e.target.value)} displayEmpty size="small" fullWidth sx={{ fontSize: '0.8rem', mt: 0.5 }}
                    renderValue={(v) => v || <span style={{ color: theme.palette.text.secondary }}>{opt.input_label || 'Select...'}</span>}>
                    {(opt.input_options || []).map((o) => <MenuItem key={o} value={o} sx={{ fontSize: '0.8rem' }}>{o}</MenuItem>)}
                  </Select>
                ) : (
                  <TextField fullWidth size="small" multiline={opt.input_type === 'text'} rows={opt.input_type === 'text' ? 2 : 1}
                    placeholder={opt.input_placeholder || opt.input_label || 'Enter details...'} value={inputValue} onChange={(e) => setInputValue(e.target.value)}
                    sx={{ mt: 0.5, '& .MuiInputBase-input': { fontSize: '0.8rem' } }} />
                )}
              </Box>
            )}
          </Box>
        ))}
      </RadioGroup>
      <Button size="small" variant="contained" disabled={!canApply} onClick={handleApply} sx={{
        mt: 1, textTransform: 'none', fontSize: '0.75rem',
        '&.Mui-disabled': { backgroundColor: theme.custom.surfaces.overlayMedium },
      }}>
        Apply Resolution
      </Button>
    </Box>
  );
}

function PendingItem({ reason, showDivider }) {
  const theme = useTheme();
  return (
    <>
      {showDivider && <Divider sx={{ my: 1.5 }} />}
      <Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, mb: 0.75 }}>
          <Typography sx={{ fontSize: '1rem' }}>{reason.icon || '⏳'}</Typography>
          <Box>
            <Typography variant="caption" sx={{ fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.3, fontSize: '0.6rem', color: theme.palette.secondary.main }}>
              {(reason.type || 'pending').replace(/_/g, ' ')}
            </Typography>
            {reason.task_title && <Typography variant="body2" sx={{ fontWeight: 500, fontSize: '0.8rem', lineHeight: 1.2 }}>{reason.task_title}</Typography>}
          </Box>
        </Box>
        <Typography variant="body2" sx={{ fontSize: '0.8rem', lineHeight: 1.4, mb: 0.5 }}>{reason.summary}</Typography>
        {reason.details && <Typography variant="caption" sx={{ color: 'text.secondary', lineHeight: 1.4, display: 'block', mb: 0.5 }}>{reason.details}</Typography>}
        {reason.pending_since && <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.65rem', display: 'block', mb: 0.5 }}>Pending since: {new Date(reason.pending_since).toLocaleDateString()} ({relativeTime(reason.pending_since)})</Typography>}
        <ResolutionOptions resolution={reason.resolution} />
      </Box>
    </>
  );
}

export default function PendingReasonPopover({ anchorEl, open, onClose, reasons }) {
  const theme = useTheme();
  const reasonList = Array.isArray(reasons) ? reasons : reasons ? [reasons] : [];
  if (reasonList.length === 0) return null;
  const title = reasonList.length === 1 ? 'Pending — Reason' : `${reasonList.length} Pending Items`;

  return (
    <Popover open={open} anchorEl={anchorEl} onClose={onClose}
      anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }} transformOrigin={{ vertical: 'top', horizontal: 'left' }}
      PaperProps={{ sx: { width: 400, maxHeight: '75vh', backgroundColor: 'background.paper', border: `1px solid ${theme.custom.surfaces.inputBorder}`, borderRadius: 2, p: 2, overflow: 'auto' } }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, mb: 1.5 }}>
        <Typography sx={{ fontSize: '1.1rem' }}>⏳</Typography>
        <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>{title}</Typography>
      </Box>
      {reasonList.map((reason, i) => <PendingItem key={i} reason={reason} showDivider={i > 0} />)}
    </Popover>
  );
}

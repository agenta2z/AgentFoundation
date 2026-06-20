import React from 'react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { TextField } from '@mui/material';

import MultiValueInput from './MultiValueInput';

const theme = createTheme();
const renderWithTheme = (ui) => render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);

afterEach(() => cleanup());

// A controlled wrapper so the array round-trips through React state like a real owner.
function Harness({ onChange }) {
  const [values, setValues] = React.useState([]);
  return (
    <MultiValueInput
      values={values}
      onChange={(next) => { setValues(next); onChange?.(next); }}
      addLabel="Add"
      renderInput={({ value, onChange: setDraft }) => (
        <TextField value={value} onChange={(e) => setDraft(e.target.value)} />
      )}
    />
  );
}

describe('MultiValueInput', () => {
  it('adds typed values to the array', () => {
    const onChange = vi.fn();
    renderWithTheme(<Harness onChange={onChange} />);

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'p1' } });
    fireEvent.click(screen.getByRole('button', { name: /add/i }));
    expect(onChange).toHaveBeenLastCalledWith(['p1']);

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'p2' } });
    fireEvent.click(screen.getByRole('button', { name: /add/i }));
    expect(onChange).toHaveBeenLastCalledWith(['p1', 'p2']);
  });

  it('removes a value via its chip delete and produces the reduced array', () => {
    const onChange = vi.fn();
    renderWithTheme(<Harness onChange={onChange} />);

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'keep' } });
    fireEvent.click(screen.getByRole('button', { name: /add/i }));
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'drop' } });
    fireEvent.click(screen.getByRole('button', { name: /add/i }));
    expect(onChange).toHaveBeenLastCalledWith(['keep', 'drop']);

    // MUI Chip delete buttons carry the "CancelIcon" test id; click the first chip's delete.
    const deleteButtons = document.querySelectorAll('.MuiChip-deleteIcon');
    expect(deleteButtons.length).toBe(2);
    fireEvent.click(deleteButtons[0]); // remove "keep"
    expect(onChange).toHaveBeenLastCalledWith(['drop']);
  });

  it('does not add duplicate or empty values', () => {
    const onChange = vi.fn();
    renderWithTheme(<Harness onChange={onChange} />);

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'dup' } });
    fireEvent.click(screen.getByRole('button', { name: /add/i }));
    expect(onChange).toHaveBeenLastCalledWith(['dup']);

    // Re-adding the same value should not change the array.
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'dup' } });
    fireEvent.click(screen.getByRole('button', { name: /add/i }));
    expect(onChange).toHaveBeenCalledTimes(1);
  });
});

import React from 'react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import { ThemeProvider, createTheme } from '@mui/material/styles';

import TextInputWidget from './TextInputWidget';

const theme = createTheme();
const renderWithTheme = (ui) => render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);

afterEach(() => cleanup());

describe('TextInputWidget — free text (unchanged)', () => {
  it('emits { content } with the trimmed typed text', () => {
    const onSubmit = vi.fn();
    const config = { input_mode: { prompt: 'What do you want?' } };
    renderWithTheme(<TextInputWidget config={config} onSubmit={onSubmit} />);

    const input = screen.getByRole('textbox');
    fireEvent.change(input, { target: { value: '  hello world  ' } });
    fireEvent.click(screen.getByRole('button', { name: /submit/i }));

    expect(onSubmit).toHaveBeenCalledWith({ content: 'hello world' });
  });

  it('renders a plain multiline textbox (not a path autocomplete) for free text', () => {
    const config = { input_mode: { prompt: 'Q' } };
    renderWithTheme(<TextInputWidget config={config} onSubmit={vi.fn()} />);
    // Free-text widget exposes a plain textbox; there is no autocomplete combobox.
    expect(screen.getByRole('textbox')).toBeTruthy();
    expect(screen.queryByRole('combobox')).toBeNull();
  });
});

describe('TextInputWidget — path compatibility fallback', () => {
  it('delegates to the path widget when expected_input_type === "path" and emits { content }', () => {
    const onSubmit = vi.fn();
    const config = {
      input_mode: { prompt: 'Pick a path', expected_input_type: 'path', prefix: '/root' },
      // no provider → plain text field inside the path widget
    };
    renderWithTheme(<TextInputWidget config={config} onSubmit={onSubmit} />);

    // The prefix adornment is a tell-tale of the delegated path widget.
    expect(screen.getByText('/root/')).toBeTruthy();

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'src/app.js' } });
    fireEvent.click(screen.getByRole('button', { name: /submit/i }));
    expect(onSubmit).toHaveBeenCalledWith({ content: 'src/app.js' });
  });
});

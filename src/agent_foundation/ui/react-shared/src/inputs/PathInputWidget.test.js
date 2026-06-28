import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { ThemeProvider, createTheme } from '@mui/material/styles';

import '../protocol/registerBuiltins';
import { getWidget, listRegisteredWidgets } from '../protocol/WidgetRegistry';
import PathInputWidget from './PathInputWidget';

const theme = createTheme();
const renderWithTheme = (ui) => render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);

afterEach(() => cleanup());

describe('path_input widget registration', () => {
  it('registers PathInputWidget under "path_input"', () => {
    expect(getWidget('path_input')).toBe(PathInputWidget);
    expect(listRegisteredWidgets()).toContain('path_input');
  });

  it('does not silently fall back to the default widget', () => {
    expect(getWidget('path_input')).not.toBe(getWidget('default'));
  });
});

describe('PathInputWidget — single value', () => {
  it('shows the prefix adornment and queries the provider with prefix + partial', async () => {
    const provider = vi.fn().mockResolvedValue([
      { name: 'features', path: '/session/root/data/features', is_dir: true },
    ]);
    const config = {
      input_mode: { prompt: 'Pick a path', expected_input_type: 'path', prefix: '/session/root' },
      pathAutocompleteProvider: provider,
    };
    renderWithTheme(<PathInputWidget config={config} onSubmit={vi.fn()} />);

    // Prefix adornment is shown (non-editable).
    expect(screen.getByText('/session/root/')).toBeTruthy();

    const input = screen.getByRole('combobox');
    fireEvent.change(input, { target: { value: 'data' } });

    await waitFor(() => expect(provider).toHaveBeenCalled());
    const call = provider.mock.calls[provider.mock.calls.length - 1][0];
    expect(call.prefix).toBe('/session/root');
    expect(call.partial).toBe('data');
  });

  it('emits { content: "<relative path>" } on submit', () => {
    const onSubmit = vi.fn();
    const provider = vi.fn().mockResolvedValue([]);
    const config = {
      input_mode: { prompt: 'Pick', expected_input_type: 'path', prefix: '/root' },
      pathAutocompleteProvider: provider,
    };
    renderWithTheme(<PathInputWidget config={config} onSubmit={onSubmit} />);

    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'src/index.js' } });
    fireEvent.click(screen.getByRole('button', { name: /submit/i }));

    expect(onSubmit).toHaveBeenCalledWith({ content: 'src/index.js' });
  });
});

describe('PathInputWidget — provider absent', () => {
  it('degrades to a plain text field (no crash) and still emits { content }', () => {
    const onSubmit = vi.fn();
    const config = {
      input_mode: { prompt: 'Pick', expected_input_type: 'path', prefix: '/root' },
      // no pathAutocompleteProvider
    };
    renderWithTheme(<PathInputWidget config={config} onSubmit={onSubmit} />);

    // A plain textbox (not an autocomplete combobox) is rendered.
    const input = screen.getByRole('textbox');
    expect(input).toBeTruthy();
    fireEvent.change(input, { target: { value: 'docs/readme.md' } });
    fireEvent.click(screen.getByRole('button', { name: /submit/i }));

    expect(onSubmit).toHaveBeenCalledWith({ content: 'docs/readme.md' });
  });
});

describe('PathInputWidget — prefill (default)', () => {
  it('pre-fills from an ABSOLUTE default, stripping the prefix; submits without retyping', () => {
    const onSubmit = vi.fn();
    const config = {
      input_mode: {
        prompt: 'Confirm the target path',
        expected_input_type: 'path',
        prefix: '/Users/zgchen/PycharmProjects/MyProjects',
        metadata: {
          expected_input_type: 'path',
          prefix: '/Users/zgchen/PycharmProjects/MyProjects',
          widget_type: 'path_input',
          default: '/Users/zgchen/PycharmProjects/MyProjects/generative_recommenders',
        },
      },
      // no provider → plain text field
    };
    renderWithTheme(<PathInputWidget config={config} onSubmit={onSubmit} />);

    // Prefix is stripped → only the relative remainder shows in the box.
    expect(screen.getByRole('textbox').value).toBe('generative_recommenders');

    // One-click confirm, no retyping.
    fireEvent.click(screen.getByRole('button', { name: /submit/i }));
    expect(onSubmit).toHaveBeenCalledWith({ content: 'generative_recommenders' });
  });

  it('accepts a RELATIVE default as-is (first-class input_mode.default)', () => {
    const config = {
      input_mode: { prompt: 'Confirm', expected_input_type: 'path', prefix: '/root', default: 'pkg/module' },
    };
    renderWithTheme(<PathInputWidget config={config} onSubmit={vi.fn()} />);
    expect(screen.getByRole('textbox').value).toBe('pkg/module');
  });
});

describe('PathInputWidget — multiple values', () => {
  it('emits { content: [...] } after adding multiple paths', () => {
    const onSubmit = vi.fn();
    const config = {
      input_mode: {
        prompt: 'Pick paths',
        expected_input_type: 'path',
        prefix: '/root',
        allow_multiple_input: true,
      },
      // no provider → plain text rows
    };
    renderWithTheme(<PathInputWidget config={config} onSubmit={onSubmit} />);

    const input = screen.getByRole('textbox');
    fireEvent.change(input, { target: { value: 'a/one' } });
    fireEvent.click(screen.getByRole('button', { name: /add path/i }));
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'b/two' } });
    fireEvent.click(screen.getByRole('button', { name: /add path/i }));

    fireEvent.click(screen.getByRole('button', { name: /^submit$/i }));
    expect(onSubmit).toHaveBeenCalledWith({ content: ['a/one', 'b/two'] });
  });
});

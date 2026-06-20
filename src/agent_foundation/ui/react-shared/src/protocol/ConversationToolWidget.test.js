import React from 'react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { ThemeProvider, createTheme } from '@mui/material/styles';

import '../protocol/registerBuiltins';
import ConversationToolWidget from './ConversationToolWidget';

const theme = createTheme();
const renderWithTheme = (ui) => render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);

afterEach(() => cleanup());

describe('ConversationToolWidget — path_input dispatch + provider threading', () => {
  it('routes metadata.widget_type "path_input" to the path widget and threads the provider', async () => {
    const provider = vi.fn().mockResolvedValue([
      { name: 'features', path: '/session/root/data/features', is_dir: true },
    ]);
    const pendingInput = {
      content: 'Choose the workflow target path.',
      inputMode: {
        mode: 'free_text',
        prompt: 'Choose the workflow target path.',
        expected_input_type: 'path',
        prefix: '/session/root',
        metadata: { widget_type: 'path_input' },
      },
    };

    renderWithTheme(
      <ConversationToolWidget
        pendingInput={pendingInput}
        onSubmit={vi.fn()}
        pathAutocompleteProvider={provider}
      />,
    );

    // Prefix adornment proves the path widget rendered.
    expect(screen.getByText('/session/root/')).toBeTruthy();

    // Typing drives the threaded provider.
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'data' } });
    await waitFor(() => expect(provider).toHaveBeenCalled());
    expect(provider.mock.calls[0][0].prefix).toBe('/session/root');
  });

  it('emits { content } from the path widget on submit', () => {
    const onSubmit = vi.fn();
    const pendingInput = {
      content: 'Pick',
      inputMode: {
        mode: 'free_text',
        prompt: 'Pick',
        expected_input_type: 'path',
        prefix: '/root',
        metadata: { widget_type: 'path_input' },
      },
    };
    // No provider → graceful plain text degradation.
    renderWithTheme(<ConversationToolWidget pendingInput={pendingInput} onSubmit={onSubmit} />);

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'a/b.txt' } });
    fireEvent.click(screen.getByRole('button', { name: /submit/i }));
    expect(onSubmit).toHaveBeenCalledWith({ content: 'a/b.txt' });
  });
});

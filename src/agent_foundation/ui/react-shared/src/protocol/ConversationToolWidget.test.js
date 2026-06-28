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

describe('ConversationToolWidget — compound: short titles + single adaptive button', () => {
  // Simple single_choice tool (no descriptions/inputs → button cards + a submit button).
  const choiceTool = (outputVar, prompt, optionLabels) => ({
    tool_type: 'single_choice',
    output_var: outputVar,
    prompt,
    input_mode: {
      mode: 'single_choice',
      options: optionLabels.map((l) => ({ label: l, value: l.toLowerCase() })),
      allow_custom: false,
    },
  });

  const compound = (tools) => ({
    content: 'preamble (shown as its own bubble)',
    inputMode: { mode: 'free_text', metadata: { compound: true, tools } },
  });

  it('multi-tab: humanized titles, Submit→Submit All, advance, single finalize payload', () => {
    const onSubmit = vi.fn();
    renderWithTheme(
      <ConversationToolWidget
        pendingInput={compound([
          choiceTool('workflow_target_path', 'Pick a path option', ['A', 'B']),
          choiceTool('workflow_mode', 'Pick a mode', ['X', 'Y']),
        ])}
        onSubmit={onSubmit}
      />,
    );

    // #1 — tabs use SHORT humanized titles, not the full prompt.
    expect(screen.getByRole('tab', { name: /Target Path/ })).toBeTruthy();
    expect(screen.getByRole('tab', { name: /Mode/ })).toBeTruthy();
    // Full question still appears in the body of the active tab.
    expect(screen.getByText('Pick a path option')).toBeTruthy();

    // #2 — one button labeled "Submit" (not all done); no separate "Submit All".
    expect(screen.getByRole('button', { name: 'Submit' })).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Submit All' })).toBeNull();

    // Commit tab 1 → advances to tab 2; nothing sent yet.
    fireEvent.click(screen.getByRole('button', { name: 'A' }));
    fireEvent.click(screen.getByRole('button', { name: 'Submit' }));
    expect(screen.getByRole('tab', { name: /✓ Target Path/ })).toBeTruthy();
    expect(onSubmit).not.toHaveBeenCalled();

    // Last tab → button flips to "Submit All"; committing it finalizes once.
    expect(screen.getByRole('button', { name: 'Submit All' })).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'X' }));
    fireEvent.click(screen.getByRole('button', { name: 'Submit All' }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(JSON.parse(onSubmit.mock.calls[0][0])).toEqual({
      workflow_target_path: { choice_index: 0 },
      workflow_mode: { choice_index: 0 },
    });
  });

  it('single-tab: no tab bar, just "Submit", finalizes with the one response', () => {
    const onSubmit = vi.fn();
    renderWithTheme(
      <ConversationToolWidget
        pendingInput={compound([choiceTool('workflow_only', 'Pick', ['A', 'B'])])}
        onSubmit={onSubmit}
      />,
    );

    expect(screen.queryByRole('tab')).toBeNull();
    expect(screen.getByRole('button', { name: 'Submit' })).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Submit All' })).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: 'A' }));
    fireEvent.click(screen.getByRole('button', { name: 'Submit' }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(JSON.parse(onSubmit.mock.calls[0][0])).toEqual({ workflow_only: { choice_index: 0 } });
  });
});

describe('ConversationToolWidget — read-only committed mode', () => {
  it('compound read-only: shows committed values, disabled, no submit buttons', () => {
    const pendingInput = {
      content: 'preamble',
      inputMode: {
        mode: 'free_text',
        metadata: {
          compound: true,
          tools: [
            {
              tool_type: 'clarification',
              output_var: 'workflow_target_path',
              prompt: 'Target path?',
              input_mode: {
                mode: 'free_text',
                expected_input_type: 'path',
                prefix: '/root',
                metadata: { widget_type: 'path_input', prefix: '/root', expected_input_type: 'path' },
              },
            },
            {
              tool_type: 'single_choice',
              output_var: 'mode',
              prompt: 'How to locate artifacts?',
              input_mode: {
                mode: 'single_choice',
                options: [{ label: 'Auto discover', value: 'auto' }, { label: 'Specify', value: 'manual' }],
                allow_custom: false,
              },
            },
          ],
        },
      },
    };
    const responseValues = {
      workflow_target_path: { content: 'generative_recommenders/' },
      mode: { choice_index: 0 },
    };

    renderWithTheme(
      <ConversationToolWidget pendingInput={pendingInput} readOnly responseValues={responseValues} onSubmit={vi.fn()} />,
    );

    // Tabs present with short titles.
    expect(screen.getByRole('tab', { name: /Target Path/ })).toBeTruthy();
    expect(screen.getByRole('tab', { name: /Mode/ })).toBeTruthy();

    // Active (path) tab shows the committed value in a DISABLED field.
    const pathInput = screen.getByRole('textbox');
    expect(pathInput.value).toBe('generative_recommenders/');
    expect(pathInput.disabled).toBe(true);

    // No submit/confirm buttons anywhere in read-only mode.
    expect(screen.queryByRole('button', { name: 'Submit' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Submit All' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Confirm Selection' })).toBeNull();

    // Switch to the choice tab → the committed option is shown, disabled.
    fireEvent.click(screen.getByRole('tab', { name: /Mode/ }));
    const chosen = screen.getByRole('button', { name: 'Auto discover' });
    expect(chosen).toBeTruthy();
    expect(chosen.disabled).toBe(true);
  });

  it('single read-only widget: shows the value disabled, no submit', () => {
    const pendingInput = {
      content: 'Pick a path',
      inputMode: {
        mode: 'free_text',
        expected_input_type: 'path',
        prefix: '/root',
        metadata: { widget_type: 'path_input', prefix: '/root', expected_input_type: 'path' },
      },
    };
    renderWithTheme(
      <ConversationToolWidget pendingInput={pendingInput} readOnly responseValues={{ content: 'a/b.txt' }} onSubmit={vi.fn()} />,
    );
    const input = screen.getByRole('textbox');
    expect(input.value).toBe('a/b.txt');
    expect(input.disabled).toBe(true);
    expect(screen.queryByRole('button', { name: 'Submit' })).toBeNull();
  });
});

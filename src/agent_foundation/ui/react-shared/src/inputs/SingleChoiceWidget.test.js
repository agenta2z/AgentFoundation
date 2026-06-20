import React from 'react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, within, cleanup } from '@testing-library/react';
import { ThemeProvider, createTheme } from '@mui/material/styles';

import SingleChoiceWidget from './SingleChoiceWidget';

const theme = createTheme();
const renderWithTheme = (ui) => render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);

afterEach(() => cleanup());

// Open the MUI Select dropdown and pick the option with the given label.
function selectOption(label) {
  fireEvent.mouseDown(screen.getByRole('combobox'));
  const listbox = within(screen.getByRole('listbox'));
  fireEvent.click(listbox.getByText(label));
}

const compositeConfig = (overrides = {}) => ({
  input_mode: {
    prompt: 'How should modeling artifacts be selected?',
    options: [
      {
        label: 'Auto-discover',
        value: 'auto_discover',
        description: 'Let the workflow inspect the target repository.',
      },
      {
        label: 'I will provide paths',
        value: 'manual_paths',
        description: 'Choose one or more artifact directories/files.',
        input: {
          name: 'workflow_modeling_artifacts_path',
          expected_input_type: 'path',
          allow_multiple_input: true,
          prefix: '/session/root',
          required: false,
        },
      },
    ],
    allow_custom: false,
  },
  ...overrides,
});

describe('SingleChoiceWidget — label-only choices (unchanged)', () => {
  it('emits { choice_index } for a simple button choice', () => {
    const onSubmit = vi.fn();
    const config = {
      input_mode: {
        prompt: 'Pick one',
        options: [{ label: 'A', value: 'a' }, { label: 'B', value: 'b' }],
        allow_custom: false,
      },
    };
    renderWithTheme(<SingleChoiceWidget config={config} onSubmit={onSubmit} />);

    fireEvent.click(screen.getByRole('button', { name: 'B' }));
    fireEvent.click(screen.getByRole('button', { name: /submit/i }));
    expect(onSubmit).toHaveBeenCalledWith({ choice_index: 1 });
  });
});

describe('SingleChoiceWidget — composite (nested input)', () => {
  it('reveals the nested input only when the input-bearing option is selected', () => {
    renderWithTheme(<SingleChoiceWidget config={compositeConfig()} onSubmit={vi.fn()} />);

    // Default selection is index 0 (Auto-discover) — no nested "Add" control.
    expect(screen.queryByRole('button', { name: /add/i })).toBeNull();

    selectOption('I will provide paths');
    // Now the multi-value path input "Add" button is present.
    expect(screen.getByRole('button', { name: /add/i })).toBeTruthy();
  });

  it('emits { choice_index, inputs: { name: [...] } } for the selected input option', () => {
    const onSubmit = vi.fn();
    renderWithTheme(<SingleChoiceWidget config={compositeConfig()} onSubmit={onSubmit} />);

    selectOption('I will provide paths');
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'data/features' } });
    fireEvent.click(screen.getByRole('button', { name: /add/i }));
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'experiments/run_42' } });
    fireEvent.click(screen.getByRole('button', { name: /add/i }));

    fireEvent.click(screen.getByRole('button', { name: /confirm selection/i }));
    expect(onSubmit).toHaveBeenCalledWith({
      choice_index: 1,
      inputs: { workflow_modeling_artifacts_path: ['data/features', 'experiments/run_42'] },
    });
  });

  it('plain option in a composite widget emits { choice_index } with NO inputs', () => {
    const onSubmit = vi.fn();
    renderWithTheme(<SingleChoiceWidget config={compositeConfig()} onSubmit={onSubmit} />);

    // Auto-discover (index 0) is already selected; submit straight away.
    fireEvent.click(screen.getByRole('button', { name: /confirm selection/i }));
    expect(onSubmit).toHaveBeenCalledWith({ choice_index: 0 });
    expect(onSubmit.mock.calls[0][0]).not.toHaveProperty('inputs');
  });

  it('switching away from an input option does NOT submit stale nested input', () => {
    const onSubmit = vi.fn();
    renderWithTheme(<SingleChoiceWidget config={compositeConfig()} onSubmit={onSubmit} />);

    // Enter a path while the manual option is selected...
    selectOption('I will provide paths');
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'stale/path' } });
    fireEvent.click(screen.getByRole('button', { name: /add/i }));

    // ...then switch back to the plain option and submit.
    selectOption('Auto-discover');
    fireEvent.click(screen.getByRole('button', { name: /confirm selection/i }));

    expect(onSubmit).toHaveBeenCalledWith({ choice_index: 0 });
    expect(onSubmit.mock.calls[0][0]).not.toHaveProperty('inputs');
  });

  it('single-value nested input emits a string (not an array)', () => {
    const onSubmit = vi.fn();
    const config = compositeConfig();
    config.input_mode.options[1].input.allow_multiple_input = false;
    renderWithTheme(<SingleChoiceWidget config={config} onSubmit={onSubmit} />);

    selectOption('I will provide paths');
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'single/path' } });
    fireEvent.click(screen.getByRole('button', { name: /confirm selection/i }));

    expect(onSubmit).toHaveBeenCalledWith({
      choice_index: 1,
      inputs: { workflow_modeling_artifacts_path: 'single/path' },
    });
  });
});

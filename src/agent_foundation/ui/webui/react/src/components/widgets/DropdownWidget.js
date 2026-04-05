/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * DropdownWidget — select dropdown with options.
 */

import React, { useState } from 'react';

export default function DropdownWidget({ config, onSubmit }) {
  const options = config?.input_mode?.options || config?.choices || [];
  const prompt = config?.input_mode?.prompt || config?.prompt || config?.title || '';
  const [selected, setSelected] = useState('');

  return (
    <div className="widget widget-dropdown">
      {prompt && <div className="widget-prompt">{prompt}</div>}
      <select
        className="widget-select"
        value={selected}
        onChange={(e) => setSelected(e.target.value)}
      >
        <option value="" disabled>Select an option...</option>
        {options.map((opt, i) => (
          <option key={i} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
      <button
        className="widget-submit"
        onClick={() => onSubmit({ choice_index: options.findIndex(o => o.value === selected) })}
        disabled={!selected}
      >
        Submit
      </button>
    </div>
  );
}

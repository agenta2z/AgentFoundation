/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * ToggleWidget — yes/no switch.
 */

import React, { useState } from 'react';

export default function ToggleWidget({ config, onSubmit }) {
  const [value, setValue] = useState(false);
  const prompt = config?.prompt || config?.title || '';

  return (
    <div className="widget widget-toggle">
      {prompt && <div className="widget-prompt">{prompt}</div>}
      <label className="widget-toggle-label">
        <input
          type="checkbox"
          checked={value}
          onChange={(e) => setValue(e.target.checked)}
        />
        <span>{value ? 'Yes' : 'No'}</span>
      </label>
      <button className="widget-submit" onClick={() => onSubmit({ value })}>
        Confirm
      </button>
    </div>
  );
}

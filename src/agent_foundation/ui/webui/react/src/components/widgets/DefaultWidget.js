/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * DefaultWidget — fallback that renders widget config as formatted JSON.
 */

import React, { useState } from 'react';

export default function DefaultWidget({ config, onSubmit }) {
  const [text, setText] = useState('');
  const prompt = config?.prompt || config?.title || 'Please provide input:';

  return (
    <div className="widget widget-default">
      <div className="widget-prompt">{prompt}</div>
      <pre className="widget-json">{JSON.stringify(config, null, 2)}</pre>
      <input
        type="text"
        className="widget-input"
        placeholder="Type your response..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && text.trim() && onSubmit({ content: text.trim() })}
      />
      <button
        className="widget-submit"
        onClick={() => onSubmit({ content: text.trim() })}
        disabled={!text.trim()}
      >
        Submit
      </button>
    </div>
  );
}

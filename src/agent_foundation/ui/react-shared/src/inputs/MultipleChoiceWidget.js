/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * MultipleChoiceWidget — checkboxes for multiple selection.
 */

import React, { useState } from 'react';

export default function MultipleChoiceWidget({ config, onSubmit, submitLabel, readOnly, value }) {
  // Read-only mode seeds the checked options + custom text from the committed value.
  const [selections, setSelections] = useState(() => (
    readOnly && value && Array.isArray(value.selections)
      ? new Set(value.selections.map((s) => s.choice_index).filter((i) => i != null))
      : new Set()
  ));
  const [customText, setCustomText] = useState(
    readOnly && value && Array.isArray(value.selections)
      ? (value.selections.find((s) => s.custom_text)?.custom_text || '')
      : '',
  );

  const options = config?.input_mode?.options || config?.options || config?.choices || [];
  const allowCustom = config?.input_mode?.allow_custom ?? config?.allow_custom ?? true;
  const prompt = config?.input_mode?.prompt || config?.prompt || config?.title || '';

  const toggleSelection = (index) => {
    const next = new Set(selections);
    if (next.has(index)) {
      next.delete(index);
    } else {
      next.add(index);
    }
    setSelections(next);
  };

  const handleSubmit = () => {
    const result = [...selections].map((i) => ({ choice_index: i }));
    if (customText.trim()) {
      result.push({ custom_text: customText.trim() });
    }
    onSubmit({ selections: result });
  };

  return (
    <div className="widget widget-multiple-choice">
      {prompt && <div className="widget-prompt">{prompt}</div>}
      <div className="widget-options">
        {options.map((opt, i) => (
          <label key={i} className={`widget-checkbox ${selections.has(i) ? 'checked' : ''}`}>
            <input
              type="checkbox"
              checked={selections.has(i)}
              onChange={() => toggleSelection(i)}
              disabled={readOnly}
            />
            <span className="option-label">{opt.label}</span>
            {opt.description && (
              <span className="option-description">{opt.description}</span>
            )}
          </label>
        ))}
      </div>
      {allowCustom && (!readOnly || customText) && (
        <input
          type="text"
          className="widget-custom-input"
          placeholder="Add custom option..."
          value={customText}
          onChange={(e) => setCustomText(e.target.value)}
          disabled={readOnly}
        />
      )}
      {!readOnly && (
        <button
          className="widget-submit"
          onClick={handleSubmit}
          disabled={selections.size === 0 && !customText.trim()}
        >
          {submitLabel || 'Submit'} ({selections.size} selected)
        </button>
      )}
    </div>
  );
}

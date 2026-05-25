/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * ToolArgumentFormWidget — compound form for collecting multiple tool arguments.
 */

import React, { useState } from 'react';
import { getWidget } from './WidgetRegistry';

export default function ToolArgumentFormWidget({ config, onSubmit }) {
  const fields = config?.fields || [];
  const title = config?.title || 'Input Required';
  const description = config?.description || '';

  const [values, setValues] = useState(() => {
    const initial = {};
    for (const field of fields) {
      initial[field.name] = field.default ?? '';
    }
    return initial;
  });

  const handleFieldChange = (name, value) => {
    setValues((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = () => {
    onSubmit({ values });
  };

  return (
    <div className="widget widget-tool-form">
      <div className="widget-form-header">
        <h4>{title}</h4>
        {description && <p className="widget-form-description">{description}</p>}
      </div>
      <div className="widget-form-fields">
        {fields.map((field) => (
          <div key={field.name} className="widget-form-field">
            <label className="widget-form-label">
              {field.label || field.name}
              {field.required && <span className="required">*</span>}
            </label>
            {field.description && (
              <span className="widget-form-field-desc">{field.description}</span>
            )}
            {field.choices ? (
              <select
                className="widget-select"
                value={values[field.name] || ''}
                onChange={(e) => handleFieldChange(field.name, e.target.value)}
              >
                <option value="">Select...</option>
                {field.choices.map((c, i) => (
                  <option key={i} value={c}>{c}</option>
                ))}
              </select>
            ) : (
              <input
                type="text"
                className="widget-input"
                value={values[field.name] || ''}
                onChange={(e) => handleFieldChange(field.name, e.target.value)}
                placeholder={field.description || `Enter ${field.name}`}
              />
            )}
          </div>
        ))}
      </div>
      <button className="widget-submit" onClick={handleSubmit}>
        Submit
      </button>
    </div>
  );
}

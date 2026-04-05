/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * CoScience Chatbot Demo - React Entry Point
 *
 * This is the main entry point for the React application.
 * It renders the App component into the DOM.
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import { createTheme } from '@mui/material/styles';
import { AppThemeProvider } from './theme/index';
import App from './App';
import './App.css';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <AppThemeProvider createThemeFn={createTheme}>
      <App />
    </AppThemeProvider>
  </React.StrictMode>
);

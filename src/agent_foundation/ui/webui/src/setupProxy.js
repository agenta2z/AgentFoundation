/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Proxy configuration for React development server.
 *
 * This file configures http-proxy-middleware to properly proxy:
 * - API requests (/api/*) to the FastAPI backend
 * - WebSocket connections (/ws/*) to the FastAPI backend
 *
 * The proxy handles SSL and IPv6 connections to the devserver backend.
 *
 * IMPORTANT: For WebSocket to work, the target must match the backend exactly.
 * In devserver setup, the backend runs on port 8087 with SSL.
 */

const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  // Get backend URL from environment or use default
  // In devserver, backend runs on port 8087 with SSL
  const backendHost = process.env.REACT_APP_BACKEND_HOST || 'localhost';
  const backendPort = process.env.REACT_APP_BACKEND_PORT || '8087';
  const backendProtocol = process.env.REACT_APP_BACKEND_PROTOCOL || 'https';
  const backendTarget = `${backendProtocol}://${backendHost}:${backendPort}`;

  // For WebSocket, we need to use ws:// or wss://
  const wsProtocol = backendProtocol === 'https' ? 'wss' : 'ws';
  const wsTarget = `${wsProtocol}://${backendHost}:${backendPort}`;

  console.log(`[Proxy] Configuring proxy to backend: ${backendTarget}`);
  console.log(`[Proxy] WebSocket target: ${wsTarget}`);

  // Proxy API requests
  app.use(
    '/api',
    createProxyMiddleware({
      target: backendTarget,
      changeOrigin: true,
      secure: false,  // Allow self-signed certificates
      logLevel: 'debug',
      // Preserve response headers for proper content serving
      onProxyRes: (proxyRes, req, res) => {
        // Ensure Content-Disposition header is forwarded
        if (proxyRes.headers['content-disposition']) {
          console.log(`[Proxy] Content-Disposition: ${proxyRes.headers['content-disposition']}`);
        }
        // Remove any headers that might cause download behavior
        delete proxyRes.headers['content-disposition-attachment'];
      },
      onProxyReq: (proxyReq, req, res) => {
        console.log(`[Proxy] API: ${req.method} ${req.url} -> ${backendTarget}`);
      },
      onError: (err, req, res) => {
        console.error(`[Proxy] API Error: ${err.message}`);
      },
    })
  );

  // Proxy WebSocket connections
  // Note: http-proxy-middleware handles WS upgrade automatically when ws: true
  const wsProxy = createProxyMiddleware({
    target: backendTarget,
    changeOrigin: true,
    secure: false,  // Allow self-signed certificates
    ws: true,  // Enable WebSocket proxying
    logLevel: 'debug',
    onProxyReq: (proxyReq, req, res) => {
      console.log(`[Proxy] WS HTTP: ${req.url} -> ${backendTarget}`);
    },
    onProxyReqWs: (proxyReq, req, socket, options, head) => {
      console.log(`[Proxy] WS Upgrade: ${req.url} -> ${backendTarget}`);
    },
    onError: (err, req, res) => {
      console.error(`[Proxy] WS Error: ${err.message}`);
    },
  });

  app.use('/ws', wsProxy);

  // Handle WebSocket upgrade manually for the dev server
  // This is needed because CRA's webpack dev server needs to know about WS routes
  app.on('upgrade', (req, socket, head) => {
    if (req.url.startsWith('/ws')) {
      console.log(`[Proxy] WS Upgrade event: ${req.url}`);
      wsProxy.upgrade(req, socket, head);
    }
  });
};

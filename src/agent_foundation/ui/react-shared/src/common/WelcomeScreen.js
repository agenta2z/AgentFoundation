/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Welcome screen component shown when chat is empty.
 */

import React from 'react';
import { Box, Typography } from '@mui/material';

/**
 * Renders the welcome screen
 * @param {object} props
 * @param {string} props.title - Application title
 * @param {string} props.logoSrc - Path to logo image
 * @param {string} props.subtitle - Subtitle text
 */
export function WelcomeScreen({
  title = 'RankEvolve',
  logoSrc = '/logo.png',
  subtitle = 'Type a message to start the demo',
}) {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        opacity: 0.5,
      }}
    >
      <Box
        component="img"
        src={logoSrc}
        alt={`${title} Logo`}
        sx={{ height: 80, mb: 2 }}
      />
      <Typography variant="h6">Welcome to {title}</Typography>
      <Typography variant="body2" sx={{ mt: 1 }}>
        {subtitle}
      </Typography>
    </Box>
  );
}

export default WelcomeScreen;

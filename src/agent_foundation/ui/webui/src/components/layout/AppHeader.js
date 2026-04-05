/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Application header/app bar component.
 */

import React from 'react';
import { AppBar, Toolbar, Typography, Box } from '@mui/material';

/**
 * Application header with logo and title
 * @param {object} props
 * @param {string} props.title - Application title
 * @param {string} props.logoSrc - Path to logo image
 */
export function AppHeader({ title = 'RankEvolve', logoSrc = '/logo.png' }) {
  return (
    <AppBar
      position="static"
      elevation={0}
      sx={{ backgroundColor: 'background.paper' }}
    >
      <Toolbar>
        <Box
          component="img"
          src={logoSrc}
          alt={`${title} Logo`}
          sx={{ height: 32, mr: 2 }}
        />
        <Typography
          variant="h5"
          sx={{ flexGrow: 1, fontSize: '1.5rem', fontWeight: 500 }}
        >
          {title}
        </Typography>
      </Toolbar>
    </AppBar>
  );
}

export default AppHeader;

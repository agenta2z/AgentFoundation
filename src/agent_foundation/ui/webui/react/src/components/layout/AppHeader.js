/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Application header/app bar component.
 */

import React from 'react';
import { AppBar, Toolbar, Typography, Box } from '@mui/material';
import { ThemeSwitcher } from '../../theme';

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
          sx={{ flexGrow: 1, fontSize: '1.6rem', fontWeight: 700 }}
        >
          {title}
        </Typography>
        <ThemeSwitcher />
      </Toolbar>
    </AppBar>
  );
}

export default AppHeader;

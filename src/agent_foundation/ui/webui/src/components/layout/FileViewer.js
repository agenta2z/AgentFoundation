/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * File viewer drawer component.
 */

import React from 'react';
import { Box, Drawer, Typography, IconButton } from '@mui/material';
import { Close as CloseIcon, Description as FileIcon } from '@mui/icons-material';
import { MarkdownRenderer } from '../common/MarkdownRenderer';
import { API_BASE } from '../../utils/api';

/**
 * File viewer drawer for displaying file contents
 * @param {object} props
 * @param {boolean} props.open - Whether drawer is open
 * @param {Function} props.onClose - Callback when drawer is closed
 * @param {string} props.fileName - Name of the file being displayed
 * @param {string} props.fileContent - File content (for non-HTML files)
 * @param {boolean} props.isHtmlFile - Whether the file is HTML
 * @param {string} props.htmlFilePath - Path for HTML file iframe src
 */
export function FileViewer({
  open,
  onClose,
  fileName,
  fileContent,
  isHtmlFile,
  htmlFilePath,
}) {
  return (
    <Drawer
      anchor="right"
      open={open}
      onClose={onClose}
      PaperProps={{
        sx: { width: { xs: '100%', sm: 600, md: 800 }, p: 2 },
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <FileIcon sx={{ mr: 1 }} />
        <Typography variant="h6" sx={{ flex: 1 }}>
          {fileName}
        </Typography>
        <IconButton onClick={onClose}>
          <CloseIcon />
        </IconButton>
      </Box>
      <Box
        sx={{
          flex: 1,
          overflow: 'auto',
          backgroundColor: isHtmlFile ? '#fff' : 'rgba(0,0,0,0.2)',
          borderRadius: 1,
          p: isHtmlFile ? 0 : 2,
          height: 'calc(100vh - 100px)',
        }}
      >
        {isHtmlFile ? (
          <iframe
            src={`${API_BASE}/experiment/static-html/${htmlFilePath}`}
            title={fileName}
            style={{
              width: '100%',
              height: '100%',
              border: 'none',
              borderRadius: '4px',
            }}
          />
        ) : (
          <MarkdownRenderer content={fileContent} />
        )}
      </Box>
    </Drawer>
  );
}

export default FileViewer;

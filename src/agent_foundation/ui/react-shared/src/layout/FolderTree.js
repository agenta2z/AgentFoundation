/**
 * FolderTree — recursive folder/file tree for the FileViewer drawer.
 *
 * Renders a collapsible directory structure. Clicking a file calls
 * onFileSelect(absolutePath). Folders expand/collapse on click.
 *
 * Props:
 *   entries        - array of {name, type, path?, size?, children?}
 *   onFileSelect   - (filePath: string) => void
 *   selectedPath   - currently selected file path (for highlight)
 *   basePath       - root directory path (prepended to build absolute paths)
 */

import React, { useState, useCallback } from 'react';
import Box from '@mui/material/Box';
import List from '@mui/material/List';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import Collapse from '@mui/material/Collapse';
import FolderIcon from '@mui/icons-material/Folder';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import DescriptionIcon from '@mui/icons-material/Description';
import ExpandLess from '@mui/icons-material/ExpandLess';
import ExpandMore from '@mui/icons-material/ExpandMore';
import { useTheme } from '@mui/material/styles';


function TreeEntry({ entry, depth, onFileSelect, selectedPath }) {
  const theme = useTheme();
  const [open, setOpen] = useState(depth < 1);
  const isDir = entry.type === 'directory';
  const isSelected = !isDir && entry.path === selectedPath;

  const handleClick = useCallback(() => {
    if (isDir) {
      setOpen(prev => !prev);
    } else if (entry.path) {
      onFileSelect(entry.path);
    }
  }, [isDir, entry.path, onFileSelect]);

  return (
    <>
      <ListItemButton
        onClick={handleClick}
        selected={isSelected}
        sx={{
          pl: 1.5 + depth * 2,
          py: 0.3,
          minHeight: 28,
          '&.Mui-selected': {
            backgroundColor: 'rgba(74,144,217,0.12)',
          },
          '&:hover': {
            backgroundColor: theme.custom?.surfaces?.overlayLight || 'rgba(255,255,255,0.04)',
          },
        }}
      >
        <ListItemIcon sx={{ minWidth: 28 }}>
          {isDir
            ? (open ? <FolderOpenIcon sx={{ fontSize: 18, color: 'warning.main' }} /> : <FolderIcon sx={{ fontSize: 18, color: 'warning.main' }} />)
            : <DescriptionIcon sx={{ fontSize: 16, color: 'text.secondary' }} />
          }
        </ListItemIcon>
        <ListItemText
          primary={entry.name}
          primaryTypographyProps={{
            fontSize: '0.82rem',
            fontFamily: 'monospace',
            fontWeight: isSelected ? 600 : 400,
            color: isDir ? 'text.primary' : 'text.secondary',
          }}
        />
        {isDir && (open ? <ExpandLess sx={{ fontSize: 16 }} /> : <ExpandMore sx={{ fontSize: 16 }} />)}
      </ListItemButton>
      {isDir && entry.children && (
        <Collapse in={open} timeout="auto" unmountOnExit>
          <List component="div" disablePadding>
            {entry.children.map((child, idx) => (
              <TreeEntry
                key={child.name + idx}
                entry={child}
                depth={depth + 1}
                onFileSelect={onFileSelect}
                selectedPath={selectedPath}
              />
            ))}
          </List>
        </Collapse>
      )}
    </>
  );
}


export function FolderTree({ entries, onFileSelect, selectedPath }) {
  if (!entries || entries.length === 0) {
    return null;
  }

  return (
    <Box sx={{ overflow: 'auto' }}>
      <List dense disablePadding>
        {entries.map((entry, idx) => (
          <TreeEntry
            key={entry.name + idx}
            entry={entry}
            depth={0}
            onFileSelect={onFileSelect}
            selectedPath={selectedPath}
          />
        ))}
      </List>
    </Box>
  );
}

export default FolderTree;

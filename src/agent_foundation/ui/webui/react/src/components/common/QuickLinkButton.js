/**
 * QuickLinkButton — icon + label link button for external resources.
 *
 * Renders as a compact outlined button with an MUI icon and label.
 * Opens the URL in a new tab.
 *
 * Props:
 *   label - string (e.g. "JIRA Board")
 *   url   - string URL
 *   icon  - string icon name (mapped to MUI icons) or React node
 */

import React from 'react';
import Button from '@mui/material/Button';
import AssignmentIcon from '@mui/icons-material/Assignment';
import ArticleIcon from '@mui/icons-material/Article';
import ChatIcon from '@mui/icons-material/Chat';
import DescriptionIcon from '@mui/icons-material/Description';
import LinkIcon from '@mui/icons-material/Link';
import LaunchIcon from '@mui/icons-material/Launch';

const ICON_MAP = {
  assignment: AssignmentIcon,
  article: ArticleIcon,
  chat: ChatIcon,
  description: DescriptionIcon,
  doc: DescriptionIcon,
  link: LinkIcon,
  jira: AssignmentIcon,
  confluence: ArticleIcon,
  slack: ChatIcon,
};

function resolveIcon(icon) {
  if (React.isValidElement(icon)) return icon;
  const IconComponent = ICON_MAP[icon?.toLowerCase()] || LaunchIcon;
  return <IconComponent sx={{ fontSize: 16 }} />;
}

export function QuickLinkButton({ label, url, icon }) {
  return (
    <Button
      variant="outlined"
      size="small"
      startIcon={resolveIcon(icon)}
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      sx={(muiTheme) => ({
        borderColor: muiTheme.custom.surfaces.inputBorder,
        color: 'text.secondary',
        fontSize: '0.75rem',
        px: 1.5,
        py: 0.5,
        '&:hover': {
          borderColor: 'primary.main',
          color: 'primary.light',
          backgroundColor: muiTheme.custom.surfaces.highlightSubtle,
        },
      })}
    >
      {label}
    </Button>
  );
}

export default QuickLinkButton;

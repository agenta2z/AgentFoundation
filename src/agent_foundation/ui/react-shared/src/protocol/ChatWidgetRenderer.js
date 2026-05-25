/**
 * ChatWidgetRenderer — registry-based renderer that maps widget types to components.
 *
 * Used by the chat session view to render structured widgets inline with messages.
 * Each widget has a `type` string and a `data` object consumed by the matching component.
 */

import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import { useTheme } from '@mui/material/styles';

import SprintProgressWidget from './SprintProgressWidget';
import WorkloadChartWidget from './WorkloadChartWidget';
import ProjectSummaryWidget from './ProjectSummaryWidget';
import TaskListWidget from './TaskListWidget';
import ApprovalWidget from './ApprovalWidget';
import ChoiceWidget from './ChoiceWidget';
import TaskAssignmentWidget from './TaskAssignmentWidget';

const WIDGET_MAP = {
  sprint_progress: SprintProgressWidget,
  workload_chart: WorkloadChartWidget,
  project_summary: ProjectSummaryWidget,
  task_list: TaskListWidget,
  approval: ApprovalWidget,
  choice: ChoiceWidget,
  task_assignment: TaskAssignmentWidget,
};

function UnknownWidget({ data, type }) {
  const theme = useTheme();
  return (
    <Box
      sx={{
        backgroundColor: theme.custom.surfaces.overlayLight,
        border: `1px solid ${theme.custom.surfaces.cardBorder}`,
        borderRadius: 2,
        p: 2,
        mt: 1.5,
      }}
    >
      <Typography variant="body2" sx={{ color: 'text.secondary', fontStyle: 'italic' }}>
        Unknown widget type: {type || 'undefined'}
      </Typography>
    </Box>
  );
}

export default function ChatWidgetRenderer({ widgets, onSubmit }) {
  if (!widgets?.length) return null;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
      {widgets.map((widget, i) => {
        const Component = WIDGET_MAP[widget.type];
        if (!Component) {
          return <UnknownWidget key={i} type={widget.type} data={widget.data} />;
        }
        // Thread onSubmit through to each widget so server can receive decisions
        const handleSubmit = onSubmit ? (result) => onSubmit(widget, result) : undefined;
        return <Component key={i} data={widget.data} onSubmit={handleSubmit} />;
      })}
    </Box>
  );
}

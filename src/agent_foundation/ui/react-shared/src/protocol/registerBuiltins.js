/**
 * Auto-registers all built-in widget types into the WidgetRegistry.
 * Imported by index.js so the registry is populated on first library import.
 */
import { registerWidget } from './WidgetRegistry';

import TextInputWidget from '../inputs/TextInputWidget';
import SingleChoiceWidget from '../inputs/SingleChoiceWidget';
import MultipleChoiceWidget from '../inputs/MultipleChoiceWidget';
import DropdownWidget from '../inputs/DropdownWidget';
import ToggleWidget from '../inputs/ToggleWidget';
import ConfirmationWidget from '../inputs/ConfirmationWidget';
import ToolArgumentFormWidget from '../inputs/ToolArgumentFormWidget';
import MultiInputWidget from '../inputs/MultiInputWidget';
import GroupedWidget from '../inputs/GroupedWidget';
import ProposalSelectionWidget from '../inputs/ProposalSelectionWidget';
import DefaultWidget from '../inputs/DefaultWidget';

registerWidget('text_input', TextInputWidget);
registerWidget('free_text', TextInputWidget);
registerWidget('single_choice', SingleChoiceWidget);
registerWidget('multiple_choice', MultipleChoiceWidget);
registerWidget('multiple_choices', MultipleChoiceWidget);
registerWidget('dropdown', DropdownWidget);
registerWidget('toggle', ToggleWidget);
registerWidget('confirmation', ConfirmationWidget);
registerWidget('press_to_continue', ConfirmationWidget);
registerWidget('tool_argument_form', ToolArgumentFormWidget);
registerWidget('multi_input', MultiInputWidget);
registerWidget('grouped', GroupedWidget);
registerWidget('proposal_selection', ProposalSelectionWidget);
registerWidget('default', DefaultWidget);

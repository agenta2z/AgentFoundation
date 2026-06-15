import './registerBuiltins';
import { getWidget, listRegisteredWidgets } from './WidgetRegistry';
import ProposalSelectionWidget from '../inputs/ProposalSelectionWidget';

describe('proposal_selection widget registration', () => {
  it('registers ProposalSelectionWidget under "proposal_selection"', () => {
    expect(getWidget('proposal_selection')).toBe(ProposalSelectionWidget);
    expect(listRegisteredWidgets()).toContain('proposal_selection');
  });

  it('does not silently fall back to the default widget', () => {
    expect(getWidget('proposal_selection')).not.toBe(getWidget('default'));
  });
});

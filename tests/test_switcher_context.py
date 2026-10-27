from contextlib import ExitStack
from unittest import mock

import pytest

from core import switcher


@pytest.mark.parametrize('cancel_after', [0, 1, 2, 3])
def test_focus_change_stops_remaining_correction_stages(cancel_after):
    calls = []
    def record(name):
        return lambda *args, **kwargs: calls.append(name)
    with ExitStack() as stack:
        for name in ('send_string_as_keys', 'send_backspaces', 'toggle_layout', 'send_vk_key'):
            stack.enter_context(mock.patch.object(switcher, name, side_effect=record(name)))
        stack.enter_context(mock.patch.object(switcher, 'get_current_layout', side_effect=['en', 'en', 'he']))
        stack.enter_context(mock.patch.object(switcher.time, 'sleep'))
        result = switcher.execute_switch('akuo', 'שלום', 'he', trigger_delimiter='\n',
                                         context_is_current=lambda: len(calls) < cancel_after)
    assert result is None
    assert len(calls) == cancel_after


def test_unconfirmed_layout_does_not_claim_target_or_submit_enter():
    with mock.patch.object(switcher, 'get_current_layout', return_value='en'), \
            mock.patch.object(switcher, 'toggle_layout'), \
            mock.patch.object(switcher, 'send_backspaces'), \
            mock.patch.object(switcher, 'send_string_as_keys') as typed, \
            mock.patch.object(switcher, 'send_vk_key') as enter, \
            mock.patch.object(switcher.time, 'sleep'):
        result = switcher.execute_switch('akuo', 'שלום', 'he', trigger_delimiter='\n',
                                         context_is_current=lambda: True)
    assert result == 'en'
    typed.assert_called_with('שלום', 'he')  # original physical keys restore the erased word
    enter.assert_not_called()


def test_success_reports_confirmed_layout():
    with mock.patch.object(switcher, 'get_current_layout', side_effect=['en', 'en', 'he']), \
            mock.patch.object(switcher, 'toggle_layout'), \
            mock.patch.object(switcher, 'send_backspaces'), \
            mock.patch.object(switcher, 'send_string_as_keys'), \
            mock.patch.object(switcher.time, 'sleep'):
        assert switcher.execute_switch('akuo', 'שלום', 'he',
                                       context_is_current=lambda: True) == 'he'

from unittest import mock
import threading

import pytest

from core import osapi
from core.hooks import HookManager, VK_A
from core.input_context import InputContext
from core.sensitivity import SensitivityManager


@pytest.fixture
def manager():
    hm = HookManager(None, SensitivityManager(), mock.Mock(), {})
    hm._running = True
    with mock.patch.object(osapi, 'is_caps_lock_on', return_value=False), \
            mock.patch.object(osapi, 'CHECK_CONTEXT_ON_KEY', True):
        yield hm


@pytest.mark.parametrize('new', [InputContext(2, 20, 'he'), InputContext(2, 20, 'en'),
                                  InputContext(1, 11, 'en')])
def test_first_key_in_new_window_drops_old_word(manager, new):
    with mock.patch.object(osapi, 'get_input_context', return_value=InputContext(1, 10, 'en')):
        manager._refresh_input_context()
        manager._on_key(VK_A, True)
    manager.history_deque.append(mock.Mock(evidence=5))
    with mock.patch.object(osapi, 'get_input_context', return_value=new):
        manager._on_key(VK_A, True)
    assert manager.buffer_active == ('ש' if new.layout == 'he' else 'a')
    assert not manager.history_deque


def test_generation_detects_return_to_same_window(manager):
    with mock.patch.object(osapi, 'get_input_context', return_value=InputContext(1, 10, 'en')):
        manager._refresh_input_context()
    generation = manager._context_generation
    with mock.patch.object(osapi, 'get_input_context', return_value=InputContext(2, 20, 'he')):
        manager._refresh_input_context()
    with mock.patch.object(osapi, 'get_input_context', return_value=InputContext(1, 10, 'en')):
        manager._refresh_input_context()
    assert manager._context_generation > generation


@pytest.mark.parametrize('context', [None, InputContext(2, 20, 'unknown')])
def test_unknown_context_passes_typing_without_analysis(manager, context):
    manager.buffer_active = 'old'
    with mock.patch.object(osapi, 'get_input_context', return_value=context):
        assert manager._on_key(VK_A, True) is False
    assert manager.buffer_active == ''


def test_focus_move_during_correction_passes_new_keys(manager):
    with mock.patch.object(osapi, 'get_input_context', return_value=InputContext(1, 10, 'en')):
        manager._refresh_input_context()
    manager.is_correcting = True
    manager._correction_target = 'he'
    with mock.patch.object(osapi, 'get_input_context', return_value=InputContext(2, 20, 'he')):
        assert manager._on_key(VK_A, True) is False
    assert not manager.pending_queue


def test_own_confirmed_layout_change_does_not_invalidate_correction(manager):
    with mock.patch.object(osapi, 'get_input_context', return_value=InputContext(1, 10, 'en')):
        manager._refresh_input_context()
    generation = manager._context_generation
    manager.is_correcting = True
    manager._correction_target = 'he'
    with mock.patch.object(osapi, 'get_input_context', return_value=InputContext(1, 10, 'he')):
        manager._refresh_input_context()
    assert manager._cached_layout == 'he'
    assert manager._context_generation == generation


def prepare_correction(manager):
    with mock.patch.object(osapi, 'get_input_context', return_value=InputContext(1, 10, 'en')):
        manager._refresh_input_context()
    manager._correction_generation = manager._context_generation
    manager._correction_target = 'he'
    manager.is_correcting = True


def test_cancelled_correction_does_not_replay_old_keys_into_new_window(manager):
    prepare_correction(manager)
    manager.pending_queue.append((VK_A, False, False))
    with mock.patch('core.hooks.execute_switch', return_value=None), \
            mock.patch.object(osapi, 'get_input_context', return_value=InputContext(2, 20, 'he')), \
            mock.patch.object(osapi, 'send_string_as_keys') as typed, \
            mock.patch.object(osapi, 'send_vk_key') as raw:
        manager._do_switch('akuo', 'שלום', 'he')
    typed.assert_not_called()
    raw.assert_not_called()
    assert not manager.is_correcting
    assert not manager.pending_queue
    assert manager.buffer_active == ''


def test_failed_layout_switch_replays_pending_keys_under_actual_layout(manager):
    prepare_correction(manager)
    manager.pending_queue.append((VK_A, False, False))
    with mock.patch('core.hooks.execute_switch', return_value='en'), \
            mock.patch.object(osapi, 'get_input_context', return_value=InputContext(1, 10, 'en')), \
            mock.patch.object(osapi, 'send_vk_key') as raw, \
            mock.patch.object(osapi, 'send_string_as_keys') as typed:
        manager._do_switch('akuo', 'שלום', 'he')
    assert manager._cached_layout == 'en'
    raw.assert_called_once_with(VK_A, shift=False)
    typed.assert_not_called()
    assert manager.buffer_active == ''


def test_success_integrates_queued_typing_after_confirmed_switch(manager):
    prepare_correction(manager)
    manager.pending_queue.append((VK_A, False, False))
    with mock.patch('core.hooks.execute_switch', return_value='he'), \
            mock.patch.object(osapi, 'get_input_context', return_value=InputContext(1, 10, 'he')), \
            mock.patch.object(osapi, 'send_string_as_keys') as typed:
        manager._do_switch('akuo', 'שלום', 'he')
    typed.assert_called_once_with('ש', 'he')
    assert manager.buffer_active == 'ש'
    assert manager._cached_layout == 'he'
    assert not manager.is_correcting


def test_slow_background_query_does_not_block_physical_typing(manager):
    entered, release, key_done = threading.Event(), threading.Event(), threading.Event()
    def slow_query():
        entered.set()
        release.wait(2)
        return None
    with mock.patch.object(osapi, 'get_input_context', side_effect=slow_query), \
            mock.patch.object(osapi, 'CHECK_CONTEXT_ON_KEY', False):
        poller = threading.Thread(target=manager._refresh_input_context)
        poller.start()
        assert entered.wait(1)
        key = threading.Thread(target=lambda: (manager._on_key(VK_A, True), key_done.set()))
        key.start()
        forwarded = key_done.wait(0.2)
        release.set()
        poller.join(1)
        key.join(1)
    assert forwarded


def test_older_background_sample_cannot_replace_newer_context(manager):
    entered, release = threading.Event(), threading.Event()
    def old_sample():
        entered.set()
        release.wait(2)
        return InputContext(1, 10, 'en')
    with mock.patch.object(osapi, 'get_input_context', side_effect=old_sample):
        poller = threading.Thread(target=manager._refresh_input_context)
        poller.start()
        assert entered.wait(1)
        with mock.patch.object(osapi, 'get_input_context', return_value=InputContext(2, 20, 'he')):
            manager._refresh_input_context()
        release.set()
        poller.join(1)
    assert manager._input_context == InputContext(2, 20, 'he')


def test_stopping_cancels_pending_correction(manager):
    prepare_correction(manager)
    manager._running = False
    with mock.patch.object(osapi, 'get_input_context', return_value=InputContext(1, 10, 'en')):
        assert manager._correction_is_current() is False


def test_superseded_sample_does_not_cancel_unchanged_correction(manager):
    prepare_correction(manager)
    entered, release = threading.Event(), threading.Event()
    results = []
    context = InputContext(1, 10, 'en')
    def old_sample():
        entered.set()
        release.wait(2)
        return context
    with mock.patch.object(osapi, 'get_input_context', side_effect=old_sample):
        guard = threading.Thread(target=lambda: results.append(manager._correction_is_current()))
        guard.start()
        assert entered.wait(1)
        with mock.patch.object(osapi, 'get_input_context', return_value=context):
            manager._refresh_input_context()
        release.set()
        guard.join(1)
    assert results == [True]

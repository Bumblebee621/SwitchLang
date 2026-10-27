from unittest import mock

from core.input_context import InputContext, read_context


def test_context_read_rejects_focus_change_during_sampling():
    identity = mock.Mock(side_effect=[(1, 10), (2, 20)])
    assert read_context(identity, lambda: 'en') is None


def test_context_includes_focused_control():
    assert read_context(lambda: (1, 10), lambda: 'he') == InputContext(1, 10, 'he')


def test_context_query_failure_is_unknown():
    assert read_context(mock.Mock(side_effect=OSError), lambda: 'en') is None


def test_missing_focus_is_unknown():
    assert read_context(lambda: (None, None), lambda: 'en') is None

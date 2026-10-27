# Per-window keyboard layout support plan

Status: implementation authorized on 2026-10-27. Initial implementation is on
`feature/per-window-layout`; GNOME full support and live platform validation
remain outstanding.

## Goal and agreed behavior

Follow the operating system's keyboard layout for the focused window on Windows
and supported Linux desktops. The user chose this approach on 2026-10-27.
SwitchLang must not maintain its own window-to-layout preferences or change the
OS setting. Continue supporting English and Hebrew only.

Example: window A uses English and window B uses Hebrew. Moving from A to B
must discard A's unfinished word and correction history, then interpret B's
typing using Hebrew. Correcting a mistake in B must not deliberately restore A's
layout or reuse A's text.

## What exists today

- `core/os_windows.py:_get_active_thread_id` and `get_current_layout` read the
  focused control's input thread. `toggle_layout` requests a change for the
  foreground window. Windows therefore already has the basic mechanism.
- `core/hooks.py:_poll_foreground_window` refreshes layout and window state every
  100 ms. It clears word buffers and history on a detected change, but reads
  layout and window separately. Immediate typing and changes during correction
  can race with those reads.
- `core/hooks.py:_do_switch` assigns the requested layout to the cache before
  confirming the result, and drains queued keys under that assumed layout.
- `core/switcher.py:execute_switch` does not check that the correction still
  belongs to the focused window. Layout switching can time out without reporting
  failure in Windows and some Linux desktop adapters.
- Hyprland reads the virtual keyboard's actual layout; X11 reads the active XKB
  group, meaning the current keyboard layout selected by the X server.
- GNOME reads the first `mru-sources` entry, a persisted recent-source list.
  This is not reliable for per-window restoration. Its adapter also returns no
  window identity, so window-change resets currently cannot work there.

These are code and upstream-source findings, not results of live Windows or
GNOME testing. Existing tests do not establish per-window compatibility.

## Can the setting be detected?

| Platform | Detection | Important limit |
| --- | --- | --- |
| Windows 8+ | `SystemParametersInfoW(SPI_GETTHREADLOCALINPUTSETTINGS, 0, &enabled, 0)`, with action `0x104E` and a `BOOL` output | The documented value describes per-thread versus session input scope. Validate its correspondence with the UI checkbox on supported Windows versions. A failed call means unknown, not disabled. |
| GNOME | Read `org.gnome.desktop.input-sources` key `per-window` | This detects the preference, not the focused window's actual active source. Missing schema or failed query means unknown. |
| Hyprland | No universal setting established by this investigation | External scripts/plugins can implement window-specific restoration. Observe actual focus/layout changes; do not infer enabled/disabled from a process name. |
| Other X11 desktops | Desktop-specific policy, no common setting established here | XKB reports keyboard state, not a universal window-layout preference. Follow observed state. |

Detection must not be a prerequisite for correct behavior: the active window and
actual active layout are authoritative, including when the preference is unknown.
Return three states for diagnostic detection: `True`, `False`, or `None`.

## Approaches considered

1. **Follow the OS automatically — selected.** Reuse native layout queries and
   reset correction context when focus/layout changes. Addresses compatibility
   without competing with the desktop's own preferences.
2. **Store layouts per window in SwitchLang — rejected.** Creates another policy
   owner and can undo the user's OS or desktop choices.
3. **Only detect the setting and warn — insufficient.** Useful as a temporary
   fallback where active context cannot be read, but does not provide support.

## Planned work

### 1. Establish platform capabilities and reproduce the failures

Files: `core/os_windows.py`, `core/os_linux.py`, `core/os_macos.py`,
`core/desktop_gnome.py`, `core/desktop_hyprland.py`, `core/desktop_x11.py`,
`tests/test_osapi.py`, new `tests/test_os_windows.py`, and
`tests/test_desktop_gnome.py`.

1. Add tests for enabled, disabled, and failed preference queries.
2. Add `get_per_window_layout_enabled() -> bool | None` to the OS interface.
3. Implement Windows and GNOME queries using the mechanisms above.
4. Return `None` on the other adapters and the existing macOS skeleton.
5. Reproduce A-English/B-Hebrew focus restoration on Windows with the preference
   enabled and disabled, including two windows of one application and modern
   Notepad's focused control.
6. Reproduce GNOME restoration while observing whether `mru-sources` changes.

Acceptance: errors are distinguishable from disabled; no preference is modified.
Do not claim a platform works based solely on mocked API tests.

### 2. Bind typing state to the observed input context

Files: `core/hooks.py`, platform adapters as needed, `core/sensitivity.py` only
if its existing window tracking needs replacement, and new
`tests/test_hooks_input_context.py`.

1. Add regression tests for A-to-B changes with different layouts, the same
   layout, and two windows of the same executable. All must clear old text and
   correction history before analyzing B's typing.
2. Introduce a coherent context snapshot containing window identity, focused
   input identity where available, actual layout, and a generation counter.
   Define the exact shared interface after Task 1 establishes what each adapter
   can reliably supply; do not use executable name as window identity.
3. Reject a sample if focus changes while it is being read. Publish valid state
   together, rather than independently updating cached fields.
4. Use focus/layout notifications where practical to invalidate stale state;
   keep polling as recovery. On Windows, evaluate a small native context check
   before analyzing a printable key. Do not put subprocesses or desktop socket
   requests inside the Windows keyboard hook.
5. If context is unknown or awaiting refresh, pass typing through and suspend
   automatic correction until fresh context is available.
6. Test rapid A-to-B-to-A transitions, typing before the next 100 ms poll,
   layout-only changes, unsupported layouts, missing focus, and query failures.

Acceptance: no previous-window word is evaluated in the new context. Measure
hook latency; event notifications alone are not proof that the first key is safe.

### 3. Keep corrections attached to their original context

Files: `core/hooks.py`, `core/switcher.py`, `core/os_windows.py`, Linux desktop
adapters, `tests/test_hooks_input_context.py`, and
`tests/test_switcher_layout_failure.py`.

1. Add tests that move focus after scheduling a correction, before erasure,
   between correction stages, and before queued-key replay.
2. Capture the originating context with each correction and revalidate before
   every destructive stage. Cancel remaining correction work on a mismatch;
   never resume old work merely because focus later returns to the same window.
3. Require layout adapters to report timeout/failure. Confirm the actual result
   before treating the requested layout as active or integrating queued text.
4. Define and test queued-key handling on cancellation: discard the old analysis
   context and never translate queued keys using an unconfirmed target layout.
   Determine during implementation whether the capture layer can preserve each
   key's original context; any unresolved ambiguity must disable correction for
   that transition rather than replay stale correction text.
5. Preserve normal corrections, Caps Lock correction, Enter submission, and
   same-window queued typing with regression tests.

Acceptance: a known focus change cancels remaining old-window correction work.
Global keyboard injection cannot make checking focus and typing atomic; document
the residual race and test it on real desktops. Do not promise that a guard can
undo text already emitted before a change becomes observable.

### 4. Resolve GNOME's active-context gap

Files initially: `core/desktop_gnome.py`, `tests/test_desktop_gnome.py`, `README.md`.
Additional integration files depend on the investigation below.

1. Test the case where the focused window switches source while `mru-sources`
   remains unchanged. The adapter must not report that persisted entry as a
   confirmed active layout.
2. Evaluate a read-only GNOME Shell integration exposing current source and
   focused-window identity, with change notifications. GNOME Shell's own source
   manager has this information, but that is not evidence of a supported public
   API for an external app. Do not assume Shell Eval or IBus's global engine
   provides the answer on supported GNOME versions.
3. If an extension is required, write a separate design for its installation,
   GNOME-version compatibility, disconnect handling, and local access boundary
   before adding that dependency.
4. Until an authoritative reader is available, disable automatic correction
   when GNOME's per-window preference is enabled or cannot be queried reliably.
   Keep physical typing working and explain the limitation in user-facing status.
5. With a confirmed reader, test opposite layouts in two windows, windows of the
   same application, focus changes during correction, and integration disconnects.

Acceptance: full GNOME per-window support requires confirmed current source and
window identity. The safe fallback is explicitly incomplete support, not a
successful implementation of the full feature.

### 5. Validate and document supported behavior

Files: `README.md` and the tests listed above.

1. Run the focused test modules, then `.venv/bin/python -m pytest tests -q`.
2. Run a live Windows matrix with the preference on/off and rapid keyboard/mouse
   focus changes, same-app windows, layout shortcuts, and failed layout requests.
3. Run a live Hyprland matrix with a window-layout manager when available; verify
   physical and virtual keyboards agree after restoration.
4. Run GNOME tests with the preference on/off and the integration absent/present.
5. Run X11 tests with a desktop that restores per-window layouts when available.
6. Document tested OS/desktop versions and remaining limits. Keep untested cases
   marked unverified. No additional settings toggle is needed for OS-following.

## Scope and constraints

- Implementation was authorized after this plan was reviewed; do not activate OS settings.
- Preserve existing uncommitted project changes and local model/corpus artifacts.
- No classifier, language scoring, or sensitivity tuning changes.
- No new macOS support or general KDE/Wayland backend in this plan.
- No new dependency until a demonstrated platform capability requires one.
- No attribution trailers in commits.

## Sources

- [Microsoft: SystemParametersInfoW](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-systemparametersinfow)
  documents the Windows input-scope query.
- [Microsoft: GetKeyboardLayout](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getkeyboardlayout)
  documents thread-specific layout reads.
- [GNOME input-source schema](https://github.com/GNOME/gsettings-desktop-schemas/blob/master/schemas/org.gnome.desktop.input-sources.gschema.xml.in)
  defines `per-window` and describes `mru-sources` as recent-source history.
- [GNOME Shell 46 keyboard manager](https://github.com/GNOME/gnome-shell/blob/gnome-46/js/ui/status/keyboard.js)
  restores sources with `activate(false)` and persists MRU only for interactive
  activations. The same relevant distinction exists in current upstream source.
- [Microsoft: SetWinEventHook](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setwineventhook)
  describes asynchronous notifications and the required message loop.
- [X.Org XKB reference](https://www.x.org/docs/XKB/XKBlib.pdf)
  describes active keyboard group queries and state notifications.

## Implementation decisions and remaining validation

- `core/input_context.py` defines immutable `InputContext(window, focus, layout)`
  samples. Native focus revisions are part of focus identity. HookManager keeps
  the correction generation separately and publishes samples in request order so
  a late background response cannot replace newer state.
- Windows queries native context before keys and tracks foreground/focus WinEvents
  on its existing hook thread. Hyprland drains its focus-event socket and queries
  active context before keys; its IPC timeout is 20 ms. X11 drains active-window
  property notifications and reads native input focus and keyboard group.
- GNOME subprocess queries run in the background outside the keyboard state lock.
  No extension dependency was added. Full GNOME per-window support requires a
  separate integration design as described in Task 4.
- Corrections validate context between stages and before each queued-key replay.
  Layout timeouts are explicit failures. Only an observed layout is cached;
  failed switches do not claim success or submit Enter.
- Cancelled corrections discard keys queued for the old context, because there
  is no safe way to replay them into a different focused window. New-context keys
  pass through. This tradeoff is documented in README.md.
- Read-only Hyprland sampling was measured on this machine with focus-event
  draining: 50 valid queries, median 0.20 ms, maximum 0.53 ms.
- Windows VM, GNOME, and X11 live per-window validation remain unverified. No
  keyboard grabs or layout changes were performed on the user's desktop.

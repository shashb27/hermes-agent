"""Windows gateway runtime detection regression tests.

Covers the fixes for NousResearch/hermes-agent#25502:

* ``_is_gateway_run_command`` only matches an actual ``gateway run`` process,
  not transient ``gateway status``/``install`` invocations, unrelated
  ``--profile`` CLI commands, or the cmd.exe wrapper that launches the gateway.
* ``_scan_gateway_pids`` skips the broad ancestor-chain exclusion on Windows
  (relying on the strict match instead), so a live ``gateway run`` process is
  never hidden.
* ``get_gateway_runtime_snapshot`` reports a Scheduled-Task-managed gateway as
  ``windows scheduled task`` (running) instead of falling through to
  ``manual process``.
"""

from types import SimpleNamespace

import pytest

import hermes_cli.gateway as gateway


# ---------------------------------------------------------------------------
# _is_gateway_run_command
# ---------------------------------------------------------------------------


class TestIsGatewayRunCommand:
    @pytest.mark.parametrize(
        "command",
        [
            "python -m hermes_cli.main gateway run",
            "python -m hermes_cli.main gateway run --replace",
            "python -m hermes_cli.main --profile work gateway run --replace",
            r"C:\venv\Scripts\pythonw.exe -m hermes_cli.main gateway run --replace",
            "/venv/bin/python /opt/hermes/hermes_cli/main.py gateway run",
            "/usr/local/bin/hermes gateway run --replace",
            "/venv/bin/python /opt/hermes/gateway/run.py",
            r"C:\opt\hermes\gateway\run.py",
        ],
    )
    def test_matches_real_gateway_runs(self, command):
        assert gateway._is_gateway_run_command(command) is True

    @pytest.mark.parametrize(
        "command",
        [
            # Other gateway subcommands are NOT long-lived gateway processes.
            "python -m hermes_cli.main gateway status",
            "python -m hermes_cli.main gateway install",
            "python -m hermes_cli.main gateway stop",
            # A --profile CLI command that isn't a gateway run at all.
            "python -m hermes_cli.main --profile work dashboard",
            "hermes gateway status",
            # The cmd.exe wrapper the Scheduled Task launches — the child python
            # process is the gateway, not this shim.
            r'C:\WINDOWS\system32\cmd.exe /c "C:\Users\me\.hermes\gateway-service\Hermes_Gateway.cmd"',
            "python -m some_other_thing",
            "",
        ],
    )
    def test_rejects_non_gateway_run_commands(self, command):
        assert gateway._is_gateway_run_command(command) is False


# ---------------------------------------------------------------------------
# _scan_gateway_pids on Windows
# ---------------------------------------------------------------------------


def _wmic_list_output(rows):
    """Render (command, pid) rows as wmic ``/FORMAT:LIST`` output."""
    blocks = []
    for command, pid in rows:
        blocks.append(f"CommandLine={command}\nProcessId={pid}\n")
    return "\n".join(blocks)


class TestWindowsScan:
    def _patch_windows_wmic(self, monkeypatch, rows):
        monkeypatch.setattr(gateway, "is_windows", lambda: True)
        monkeypatch.setattr(gateway.shutil, "which", lambda name: "wmic" if name == "wmic" else None)

        output = _wmic_list_output(rows)

        def fake_run(cmd, **kwargs):
            assert cmd[:2] == ["wmic", "process"], f"unexpected command: {cmd}"
            return SimpleNamespace(returncode=0, stdout=output, stderr="")

        monkeypatch.setattr(gateway.subprocess, "run", fake_run)

    def test_matches_only_gateway_run_process(self, monkeypatch):
        """The transient ``gateway status`` and the cmd.exe wrapper are ignored;
        only the real ``gateway run`` python process is returned."""
        self._patch_windows_wmic(
            monkeypatch,
            rows=[
                ("python -m hermes_cli.main gateway run --replace", 12345),
                ("python -m hermes_cli.main gateway status", 22222),
                (
                    r'C:\WINDOWS\system32\cmd.exe /c "C:\Users\me\.hermes\gateway-service\Hermes_Gateway.cmd"',
                    33333,
                ),
                ("python -m some_other_thing", 44444),
            ],
        )

        pids = gateway._scan_gateway_pids(set(), all_profiles=True)

        assert pids == [12345]

    def test_skips_broad_ancestor_exclusion_on_windows(self, monkeypatch):
        """On Windows the ancestor-chain exclusion is not applied, so a live
        ``gateway run`` process is returned even if it appears in the ancestor
        set that would exclude it on POSIX."""
        ancestor_calls = []

        def _boom():
            ancestor_calls.append(True)
            return {12345}

        monkeypatch.setattr(gateway, "_get_ancestor_pids", _boom)
        self._patch_windows_wmic(
            monkeypatch,
            rows=[("python -m hermes_cli.main gateway run --replace", 12345)],
        )

        pids = gateway._scan_gateway_pids(set(), all_profiles=True)

        assert pids == [12345]
        assert ancestor_calls == [], "ancestor walk must be skipped on Windows"

    def test_posix_still_applies_ancestor_exclusion(self, monkeypatch):
        """Regression guard: POSIX behaviour is unchanged — an ancestor
        ``gateway run`` PID is still excluded (see #13242)."""
        monkeypatch.setattr(gateway, "is_windows", lambda: False)
        monkeypatch.setattr(gateway, "_get_ancestor_pids", lambda: {12345})
        monkeypatch.setattr(gateway.os.path, "isdir", lambda path: False)

        ps_output = "12345 python -m hermes_cli.main gateway run\n"

        def fake_run(cmd, **kwargs):
            return SimpleNamespace(returncode=0, stdout=ps_output, stderr="")

        monkeypatch.setattr(gateway.subprocess, "run", fake_run)

        pids = gateway._scan_gateway_pids(set(), all_profiles=True)

        assert pids == []


# ---------------------------------------------------------------------------
# get_gateway_runtime_snapshot Windows branch
# ---------------------------------------------------------------------------


class TestWindowsRuntimeSnapshot:
    def _force_windows(self, monkeypatch, pids):
        # Steer get_gateway_runtime_snapshot straight to the Windows branch on a
        # non-Windows test host.
        monkeypatch.setattr(gateway, "find_gateway_pids", lambda: list(pids))
        monkeypatch.setattr(gateway, "is_termux", lambda: False)
        monkeypatch.setattr(gateway, "is_linux", lambda: False)
        monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
        monkeypatch.setattr(gateway, "is_macos", lambda: False)
        monkeypatch.setattr(gateway, "is_windows", lambda: True)

    def _patch_gateway_windows(self, monkeypatch, *, task_registered, startup_installed, status):
        from hermes_cli import gateway_windows

        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: task_registered)
        monkeypatch.setattr(gateway_windows, "is_startup_entry_installed", lambda: startup_installed)
        monkeypatch.setattr(
            gateway_windows,
            "query_task_status",
            lambda: ({"status": status} if status is not None else {}),
        )

    def test_scheduled_task_running(self, monkeypatch):
        self._force_windows(monkeypatch, pids=[4242])
        self._patch_gateway_windows(
            monkeypatch, task_registered=True, startup_installed=False, status="Running"
        )

        snap = gateway.get_gateway_runtime_snapshot()

        assert snap.manager == "windows scheduled task"
        assert snap.service_installed is True
        assert snap.service_running is True
        assert snap.running is True
        assert snap.gateway_pids == (4242,)
        assert snap.has_process_service_mismatch is False

    def test_scheduled_task_ready_but_process_alive(self, monkeypatch):
        """Task shows ``Ready`` yet a gateway process is alive — still reported
        as running (never a false ``stopped``) and not a mismatch."""
        self._force_windows(monkeypatch, pids=[4242])
        self._patch_gateway_windows(
            monkeypatch, task_registered=True, startup_installed=False, status="Ready"
        )

        snap = gateway.get_gateway_runtime_snapshot()

        assert snap.manager == "windows scheduled task"
        assert snap.service_running is True
        assert snap.running is True
        assert snap.has_process_service_mismatch is False

    def test_scheduled_task_installed_but_stopped(self, monkeypatch):
        self._force_windows(monkeypatch, pids=[])
        self._patch_gateway_windows(
            monkeypatch, task_registered=True, startup_installed=False, status="Ready"
        )

        snap = gateway.get_gateway_runtime_snapshot()

        assert snap.manager == "windows scheduled task"
        assert snap.service_installed is True
        assert snap.service_running is False
        assert snap.running is False

    def test_status_unavailable_falls_back_to_process(self, monkeypatch):
        """When schtasks status can't be read, a live process still reads as
        running rather than stopped."""
        self._force_windows(monkeypatch, pids=[99])
        self._patch_gateway_windows(
            monkeypatch, task_registered=True, startup_installed=False, status=None
        )

        snap = gateway.get_gateway_runtime_snapshot()

        assert snap.service_running is True
        assert snap.running is True

    def test_startup_folder_fallback(self, monkeypatch):
        self._force_windows(monkeypatch, pids=[7])
        self._patch_gateway_windows(
            monkeypatch, task_registered=False, startup_installed=True, status=None
        )

        snap = gateway.get_gateway_runtime_snapshot()

        assert snap.manager == "windows startup item"
        assert snap.service_installed is True
        assert snap.service_running is True

    def test_no_service_reports_manual_process(self, monkeypatch):
        self._force_windows(monkeypatch, pids=[])
        self._patch_gateway_windows(
            monkeypatch, task_registered=False, startup_installed=False, status=None
        )

        snap = gateway.get_gateway_runtime_snapshot()

        assert snap.manager == "manual process"
        assert snap.service_installed is False
        assert snap.running is False

    def test_probe_failure_degrades_to_manual(self, monkeypatch):
        """A crash while probing the Windows backend must not break status."""
        from hermes_cli import gateway_windows

        self._force_windows(monkeypatch, pids=[321])

        def _boom():
            raise OSError("schtasks exploded")

        monkeypatch.setattr(gateway_windows, "is_task_registered", _boom)

        snap = gateway.get_gateway_runtime_snapshot()

        assert snap.manager == "manual process"
        assert snap.gateway_pids == (321,)

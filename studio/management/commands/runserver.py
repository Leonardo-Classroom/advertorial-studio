"""`runserver` with this project's default port, and a reloader that survives.

Subclasses the staticfiles variant rather than the core one so static files
are still served in DEBUG — plain `django.core.management.commands.runserver`
would silently stop serving them.

An explicit address still wins: `python manage.py runserver 8000` works as
usual; only the no-argument default changes.
"""
import sys

from django.conf import settings
from django.contrib.staticfiles.management.commands.runserver import (
    Command as StaticfilesRunserverCommand,
)


def _survive_reloader_scans() -> None:
    """Stop a failed file scan from taking the whole server down with it.

    2026-08-11, mid-generation, the dev server died on this:

        File ".../django/utils/autoreload.py", line 119, in iter_all_python_module_files
        File ".../posixpath.py", line 83, in join
        UnboundLocalError: local variable 'b' referenced before assignment

    That error is not possible from the source — `b` is the loop variable and
    the reference is the first statement of the loop body; the bytecode was
    checked and is correct. It is an interpreter-level fault, and this process
    is where one would show up: it holds torch/CUDA in memory while the
    reloader thread resolves tens of thousands of module paths across a drvfs
    mount, several times a second.

    The cause is not fixable from here. The consequence is: the run in flight
    (`draft 556`) died with the process and sat at 生成中 until the staleness
    sweep noticed, because worker threads do not outlive their server.

    So the scan is made best-effort, which is all it ever was. A tick that
    cannot enumerate files reuses the previous snapshot instead of raising:
    auto-reload misses that one tick, and the server — plus whatever it is
    generating — stays up. Repeated failures are reported once so a reloader
    that has genuinely stopped working does not do so silently.
    """
    from django.utils import autoreload

    original = autoreload.iter_all_python_module_files
    state = {"last": frozenset(), "failures": 0, "warned": False}

    def resilient():
        try:
            files = original()
        except Exception as exc:  # noqa: BLE001 - deliberately broad; see docstring
            state["failures"] += 1
            if not state["warned"]:
                state["warned"] = True
                print(f"\n[runserver] 自動重載掃描失敗（{type(exc).__name__}: {exc}）。"
                      f"\n[runserver] 已沿用上一次的檔案快照繼續執行——伺服器與正在產的稿"
                      f"不受影響，但這一輪的檔案變更不會觸發重載。"
                      f"\n[runserver] 若之後改了程式沒有自動重載，請手動重啟。\n",
                      file=sys.stderr)
            return state["last"]
        state["last"] = files
        state["failures"] = 0
        return files

    autoreload.iter_all_python_module_files = resilient


class Command(StaticfilesRunserverCommand):
    default_port = str(getattr(settings, "RUNSERVER_PORT", "5860"))

    def run(self, **options):
        # Only meaningful when the reloader is actually in use; harmless
        # otherwise, and patching before `super()` keeps it in place for the
        # child process the reloader spawns.
        _survive_reloader_scans()
        super().run(**options)

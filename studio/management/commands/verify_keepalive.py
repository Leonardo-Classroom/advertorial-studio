"""Check that the「本地模型閒置卸載分鐘數」setting really controls VRAM residency.

    python manage.py verify_keepalive

Ollama honours `keep_alive` only on its native `/api/*` endpoints — the
OpenAI-compatible `/v1/responses` and `/v1/chat/completions` paths silently
drop it and leave the server default of 5 minutes in place. That failure is
invisible from the app's side: the request succeeds, the field is well-formed,
and nothing reports that the receiver ignored it. `core.llm._touch_keep_alive`
works around it with a separate native call; this command is how we know that
workaround is still working.

It drives the real code path (`core.llm.complete`) at two different settings,
reads `/api/ps` after each, and restores the original value on the way out.
"""
import re
import time
from datetime import datetime, timezone

import httpx
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from core import llm
from studio.models import SiteSettings

# Seconds of slack between issuing the call and reading back the deadline.
TOLERANCE = 25


def _native_root() -> str:
    """The native API root beside the OpenAI-compatible `.../v1` base URL."""
    return settings.LOCAL_LLM_BASE_URL.rstrip("/").removesuffix("/v1")


def _seconds_left():
    """How long until the loaded model is unloaded, per `/api/ps`.

    Ollama stamps `expires_at` with nanoseconds (9 decimal places) and
    `datetime.fromisoformat` only parses microseconds, so trim to 6.
    """
    models = httpx.get(f"{_native_root()}/api/ps", timeout=15).json().get("models") or []
    if not models:
        return None, None
    stamp = re.sub(r"\.(\d{6})\d+", r".\1",
                   models[0]["expires_at"].replace("Z", "+00:00"))
    delta = (datetime.fromisoformat(stamp) - datetime.now(timezone.utc)).total_seconds()
    return delta, models[0]["name"]


class Command(BaseCommand):
    help = "驗證「本地模型閒置卸載分鐘數」確實控制模型在 VRAM 的停留時間"

    def add_arguments(self, parser):
        parser.add_argument(
            "--minutes", type=int, nargs="+", default=[8, 3],
            help="要試的分鐘數（預設 8 3，用兩個不同值才能證明設定有被讀到）")

    def handle(self, *args, **options):
        try:
            version = httpx.get(f"{_native_root()}/api/version", timeout=5).json().get("version")
        except Exception as exc:  # noqa: BLE001
            raise CommandError(f"Ollama 連不上（{exc}）——請先啟動（./restart_ollama.sh）。")

        site = SiteSettings.load()
        if site.llm_backend != "local":
            raise CommandError("目前後端不是本地模型，這個設定不會生效——先切到本地再測。")
        self.stdout.write(f"Ollama {version}｜目前設定 {site.ollama_idle_unload_minutes} 分鐘")

        original = site.ollama_idle_unload_minutes
        failures = []
        try:
            for minutes in options["minutes"]:
                site.ollama_idle_unload_minutes = minutes
                site.save(update_fields=["ollama_idle_unload_minutes"])
                started = time.time()
                reply = llm.complete("只回一個字。", "說好", max_output_tokens=8)
                elapsed = time.time() - started
                time.sleep(1)
                delta, name = _seconds_left()
                if delta is None:
                    failures.append(f"{minutes} 分鐘：模型已不在 VRAM 中")
                    self.stdout.write(self.style.ERROR(
                        f"✗ 設定 {minutes} 分鐘 -> /api/ps 是空的"))
                    continue
                if abs(delta - minutes * 60) > TOLERANCE:
                    failures.append(f"{minutes} 分鐘：實得 {delta:.0f}s")
                    style = self.style.ERROR
                    mark = "✗"
                else:
                    style = self.style.SUCCESS
                    mark = "✓"
                self.stdout.write(style(
                    f"{mark} 設定 {minutes} 分鐘（{minutes * 60}s）-> 實際剩 {delta:.0f}s"
                    f"｜{name}｜呼叫 {elapsed:.1f}s 回 {reply[:10]!r}"))
        finally:
            site.ollama_idle_unload_minutes = original
            site.save(update_fields=["ollama_idle_unload_minutes"])
            self.stdout.write(f"已還原為 {original} 分鐘")

        if failures:
            raise CommandError("未通過：" + "；".join(failures))
        self.stdout.write(self.style.SUCCESS(
            "通過：設定的分鐘數確實決定模型在 VRAM 的停留時間。"))

"""Embed the corpus into the vector index.

    python manage.py build_index --outlet COOL-STYLE
    python manage.py build_index --outlet GQ --limit-per-author 300
    python manage.py build_index --rebuild

Incremental by default: articles already in the index are skipped, so this can
be run repeatedly as the corpus grows.
"""
from __future__ import annotations

import sys
import time

from django.core.management.base import BaseCommand

from corpus.services import index


class Command(BaseCommand):
    help = "為語料文章建立向量索引（Cohere embed-v-4-0）"

    def add_arguments(self, parser):
        parser.add_argument("--outlet", default=None, help="只索引單一媒體，例如 GQ")
        parser.add_argument("--limit-per-author", type=int, default=0,
                            help="每位作者最多索引幾篇（取最新，0=不限）。用來先建小規模索引試跑。")
        parser.add_argument("--rebuild", action="store_true", help="捨棄既有索引重建")

    def handle(self, *args, **opts):
        started = time.time()
        last = [0.0]

        def progress(done, total):
            now = time.time()
            if now - last[0] < 1.0 and done < total:
                return
            last[0] = now
            rate = done / max(now - started, 0.001)
            eta = (total - done) / rate if rate else 0
            self.stdout.write(
                f"  嵌入中 {done}/{total} ({done / total:.1%})  "
                f"{rate:.0f} 筆/秒  預估剩餘 {eta / 60:.1f} 分",
                ending="\r",
            )
            self.stdout.flush()

        try:
            record = index.build(
                outlet=opts["outlet"],
                limit_per_author=opts["limit_per_author"],
                rebuild=opts["rebuild"],
                progress=progress,
            )
        except Exception as exc:  # noqa: BLE001 - surface the real cause to the operator
            self.stderr.write(self.style.ERROR(f"\n索引建立失敗：{exc}"))
            sys.exit(1)

        elapsed = time.time() - started
        self.stdout.write("\n" + self.style.SUCCESS(
            f"索引完成：{record.vector_count} 筆向量 / {record.dimensions} 維 "
            f"/ {record.model}，耗時 {elapsed / 60:.1f} 分"
        ))

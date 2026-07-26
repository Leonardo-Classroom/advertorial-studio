"""Load the crawled GQ / COOL-STYLE corpus into the database.

    python manage.py ingest_corpus --outlet COOL-STYLE
    python manage.py ingest_corpus                      # everything under CORPUS_ROOT
    python manage.py ingest_corpus --dry-run --sample 20  # inspect cleaning first

Re-running is safe: articles are keyed by content hash, so unchanged files are
skipped rather than duplicated.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils.text import slugify

from corpus.models import Article, Author, Outlet
from corpus.services import ingest


class Command(BaseCommand):
    help = "掃描語料資料夾，解析並清洗文章後寫入資料庫"

    def add_arguments(self, parser):
        parser.add_argument("--root", default=None, help="語料根目錄（預設取 settings.CORPUS_ROOT）")
        parser.add_argument("--outlet", default=None, help="只處理單一媒體資料夾，例如 GQ")
        parser.add_argument("--target", action="store_true",
                            help="只處理模仿目標媒體（GQ 與 COOL-STYLE）")
        parser.add_argument("--limit", type=int, default=0, help="最多處理幾篇（0=不限）")
        parser.add_argument("--min-chars", type=int, default=200,
                            help="內文低於此字數者視為無效（預設 200）")
        parser.add_argument("--dry-run", action="store_true", help="只解析不寫入")
        parser.add_argument("--sample", type=int, default=0,
                            help="dry-run 時印出前 N 篇的清洗結果供人工檢查")

    def handle(self, *args, **opts):
        from django.conf import settings

        root = Path(opts["root"]) if opts["root"] else settings.CORPUS_ROOT
        if not root.exists():
            self.stderr.write(self.style.ERROR(f"語料根目錄不存在：{root}"))
            sys.exit(1)

        targets = ["GQ", "COOL-STYLE"] if opts["target"] else [opts["outlet"]]
        stats = Counter()
        dropped_examples: Counter = Counter()
        shown = 0

        for outlet_name in targets:
            files = ingest.iter_article_files(root, outlet_name)
            outlet_cache: dict[str, Outlet] = {}
            author_cache: dict[tuple[int, str], Author] = {}
            pending: list[Article] = []

            for path in files:
                if opts["limit"] and stats["parsed"] >= opts["limit"]:
                    break
                stats["seen"] += 1
                parsed = ingest.parse_file(path)
                if parsed is None:
                    stats["unparseable"] += 1
                    continue
                if parsed.char_count < opts["min_chars"]:
                    stats["too_short"] += 1
                    continue
                stats["parsed"] += 1
                for line in parsed.dropped_lines:
                    dropped_examples[line[:60]] += 1

                if opts["dry_run"]:
                    if shown < opts["sample"]:
                        shown += 1
                        self.stdout.write(self.style.HTTP_INFO(f"\n--- {path.name} ---"))
                        self.stdout.write(f"標題: {parsed.title}")
                        self.stdout.write(f"作者: {parsed.author} / 字數: {parsed.char_count}")
                        self.stdout.write(f"內文前 200 字: {parsed.body[:200]}")
                        self.stdout.write(f"已移除 {len(parsed.dropped_lines)} 行雜訊")
                    continue

                outlet = outlet_cache.get(parsed.outlet)
                if outlet is None:
                    outlet, _ = Outlet.objects.get_or_create(
                        name=parsed.outlet,
                        defaults={
                            "slug": slugify(parsed.outlet) or parsed.outlet.lower(),
                            "is_target": parsed.outlet in {"GQ", "COOL-STYLE"},
                        },
                    )
                    outlet_cache[parsed.outlet] = outlet

                key = (outlet.id, parsed.author)
                author = author_cache.get(key)
                if author is None:
                    author, _ = Author.objects.get_or_create(outlet=outlet, name=parsed.author)
                    author_cache[key] = author

                pending.append(Article(
                    outlet=outlet, author=author, title=parsed.title[:512],
                    url=parsed.url, published_on=parsed.published_on,
                    body=parsed.body, char_count=parsed.char_count,
                    content_hash=parsed.content_hash, source_path=parsed.source_path,
                ))

                if len(pending) >= 500:
                    stats["written"] += self._flush(pending)
                    pending = []
                    self.stdout.write(f"  ... 已寫入 {stats['written']} 篇", ending="\r")

            if pending and not opts["dry_run"]:
                stats["written"] += self._flush(pending)

        if not opts["dry_run"]:
            self._refresh_counts()

        self.stdout.write("\n" + self.style.SUCCESS("=== 匯入結果 ==="))
        for k in ["seen", "parsed", "written", "unparseable", "too_short"]:
            self.stdout.write(f"  {k:12s} {stats[k]}")
        stats_dup = stats["parsed"] - stats["written"] if not opts["dry_run"] else 0
        if stats_dup > 0:
            self.stdout.write(f"  {'duplicate':12s} {stats_dup}（內容重複，已跳過）")

        if dropped_examples:
            self.stdout.write("\n移除的雜訊行 Top 10（供檢查清洗規則是否誤刪）：")
            for line, n in dropped_examples.most_common(10):
                self.stdout.write(f"  {n:6d}x  {line}")

    def _flush(self, pending: list[Article]) -> int:
        """Insert a batch, letting the DB drop content-hash duplicates."""
        with transaction.atomic():
            existing = set(
                Article.objects.filter(
                    content_hash__in=[a.content_hash for a in pending]
                ).values_list("content_hash", flat=True)
            )
            fresh, seen = [], set()
            for a in pending:
                if a.content_hash in existing or a.content_hash in seen:
                    continue
                seen.add(a.content_hash)
                fresh.append(a)
            Article.objects.bulk_create(fresh, batch_size=500)
        return len(fresh)

    def _refresh_counts(self):
        from django.db.models import Count
        for author in Author.objects.annotate(n=Count("articles")):
            if author.article_count != author.n:
                Author.objects.filter(pk=author.pk).update(article_count=author.n)

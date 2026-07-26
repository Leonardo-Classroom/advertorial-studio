"""Induce a style guide for an outlet, or for one author.

    python manage.py extract_style_guide --outlet COOL-STYLE
    python manage.py extract_style_guide --outlet GQ --author "Daniel Hsu"
    python manage.py extract_style_guide --outlet COOL-STYLE --all-authors --min-articles 500
"""
from __future__ import annotations

import sys

from django.core.management.base import BaseCommand

from corpus.models import Author, Outlet
from corpus.services import styleguide


class Command(BaseCommand):
    help = "分析語料並產生可編輯的風格指南"

    def add_arguments(self, parser):
        parser.add_argument("--outlet", required=True, help="媒體名稱，例如 GQ")
        parser.add_argument("--author", default=None, help="作者名稱；不給則產生全站指南")
        parser.add_argument("--all-authors", action="store_true",
                            help="為樣本數足夠的作者各產生一份")
        parser.add_argument("--min-articles", type=int, default=500,
                            help="--all-authors 時的樣本數門檻（預設 500）")
        parser.add_argument("--sample-size", type=int, default=24,
                            help="送進模型分析的文章篇數")

    def handle(self, *args, **opts):
        try:
            outlet = Outlet.objects.get(name=opts["outlet"])
        except Outlet.DoesNotExist:
            self.stderr.write(self.style.ERROR(
                f"找不到媒體「{opts['outlet']}」。已匯入的有："
                + "、".join(Outlet.objects.values_list("name", flat=True))
            ))
            sys.exit(1)

        targets: list[Author | None] = []
        if opts["all_authors"]:
            authors = Author.objects.filter(
                outlet=outlet, article_count__gte=opts["min_articles"]
            ).order_by("-article_count")
            if not authors:
                self.stderr.write(self.style.WARNING(
                    f"{outlet.name} 沒有作者達到 {opts['min_articles']} 篇門檻。"
                ))
            targets = list(authors)
        elif opts["author"]:
            try:
                targets = [Author.objects.get(outlet=outlet, name=opts["author"])]
            except Author.DoesNotExist:
                self.stderr.write(self.style.ERROR(f"找不到作者「{opts['author']}」"))
                sys.exit(1)
        else:
            targets = [None]

        for author in targets:
            scope = f"{outlet.name} / {author.name}" if author else f"{outlet.name}（全站）"
            self.stdout.write(f"分析中：{scope} …")
            try:
                guide = styleguide.induce(
                    outlet=outlet, author=author, sample_size=opts["sample_size"]
                )
            except Exception as exc:  # noqa: BLE001 - report and continue to next author
                self.stderr.write(self.style.ERROR(f"  失敗：{exc}"))
                continue

            if author:
                conf = author.style_confidence
                if conf == "low":
                    self.stdout.write(self.style.WARNING(
                        f"  ⚠ 此作者僅 {author.article_count} 篇，風格信心度低，"
                        "建議改用全站指南或高產作者。"
                    ))
            self.stdout.write(self.style.SUCCESS(
                f"  完成 → StyleGuide #{guide.pk}（取樣 {guide.sample_size} 篇）"
            ))

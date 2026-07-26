from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Q
from django.shortcuts import get_object_or_404, redirect, render

from accounts.decorators import staff_required
from corpus.models import Article, Author, EmbeddingIndex, Outlet, StyleGuide


@staff_required
def articles(request):
    qs = Article.objects.select_related("author", "outlet")
    outlet_id = request.GET.get("outlet") or ""
    author_id = request.GET.get("author") or ""
    q = (request.GET.get("q") or "").strip()
    indexed_only = request.GET.get("indexed") == "1"

    if outlet_id:
        qs = qs.filter(outlet_id=outlet_id)
    if author_id:
        qs = qs.filter(author_id=author_id)
    if q:
        qs = qs.filter(Q(title__icontains=q) | Q(body__icontains=q))
    if indexed_only:
        qs = qs.exclude(vector_row__isnull=True)

    page = Paginator(qs, 40).get_page(request.GET.get("page"))
    author_qs = Author.objects.select_related("outlet").order_by("-article_count")
    if outlet_id:
        author_qs = author_qs.filter(outlet_id=outlet_id)

    return render(request, "corpus/articles.html", {
        "section": "corpus",
        "page": page,
        "outlets": Outlet.objects.all(),
        "authors": author_qs[:200],
        "sel_outlet": outlet_id,
        "sel_author": author_id,
        "q": q,
        "indexed_only": indexed_only,
    })


@staff_required
def article_detail(request, pk):
    article = get_object_or_404(Article.objects.select_related("author", "outlet"), pk=pk)
    return render(request, "corpus/article_detail.html", {
        "section": "corpus", "article": article,
    })


@staff_required
def authors(request):
    outlet_id = request.GET.get("outlet") or ""
    qs = Author.objects.select_related("outlet").order_by("-article_count")
    if outlet_id:
        qs = qs.filter(outlet_id=outlet_id)
    return render(request, "corpus/authors.html", {
        "section": "corpus",
        "page": Paginator(qs, 60).get_page(request.GET.get("page")),
        "outlets": Outlet.objects.all(),
        "sel_outlet": outlet_id,
    })


@staff_required
def guides(request):
    return render(request, "corpus/guides.html", {
        "section": "guides",
        "guides": StyleGuide.objects.select_related("outlet", "author").all(),
        "index": EmbeddingIndex.objects.filter(name="articles").first(),
    })


@staff_required
def guide_detail(request, pk):
    guide = get_object_or_404(StyleGuide.objects.select_related("outlet", "author"), pk=pk)
    if request.method == "POST":
        guide.content = request.POST.get("content", guide.content)
        guide.is_active = request.POST.get("is_active") == "on"
        guide.save(update_fields=["content", "is_active", "updated_at"])
        messages.success(request, "風格指南已更新，生成時會直接採用你編輯後的版本。")
        return redirect("corpus:guide_detail", pk=guide.pk)

    sections = guide.sections or {}
    measured = sections.get("measured") or {}
    # Punctuation keys are full-width symbols, which Django's template variable
    # syntax cannot address by attribute lookup — flatten them here instead.
    punctuation = list((measured.get("punctuation_per_100_chars") or {}).items())
    body_rows = list((measured.get("body") or {}).items())

    return render(request, "corpus/guide_detail.html", {
        "section": "guides",
        "guide": guide,
        "measured": measured,
        "punctuation": punctuation,
        "body_rows": body_rows,
        "terms": sections.get("distinctive_terms") or [],
    })

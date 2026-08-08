from django.contrib import messages
from django.db.models import Q
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render

from accounts.decorators import staff_required
from core.pagination import paginate, query_string
from corpus.models import Article, Author, EmbeddingIndex, Outlet, StyleGuide


@staff_required
def articles(request):
    # No `select_related` here — `paginate(related=...)` applies it after the
    # slice, which is what keeps a deep page from resolving the join for every
    # row it skips. See `core.pagination.paginate`.
    qs = Article.objects.all()
    outlet_id = request.GET.get("outlet") or ""
    author_id = request.GET.get("author") or ""
    q = (request.GET.get("q") or "").strip()

    if outlet_id:
        qs = qs.filter(outlet_id=outlet_id)
    if author_id:
        qs = qs.filter(author_id=author_id)
    if q:
        qs = qs.filter(Q(title__icontains=q) | Q(body__icontains=q))

    author_qs = Author.objects.select_related("outlet").order_by("-article_count")
    if outlet_id:
        author_qs = author_qs.filter(outlet_id=outlet_id)

    return render(request, "corpus/articles.html", {
        "section": "corpus",
        **paginate(request, qs, 40, related=("author", "outlet"),
                   q=q, outlet=outlet_id, author=author_id),
        "outlets": Outlet.objects.all(),
        "authors": author_qs[:200],
        "sel_outlet": outlet_id,
        "sel_author": author_id,
        "q": q,
    })


@staff_required
def article_detail(request, pk):
    article = get_object_or_404(Article.objects.select_related("author", "outlet"), pk=pk)
    return render(request, "corpus/article_detail.html", {
        "section": "corpus", "article": article,
    })


# `style_confidence` is derived from `article_count` alone (>=500 high,
# >=100 medium), so sorting by it *is* sorting by the count — the two headers
# deliberately map to the same field rather than one of them doing nothing.
AUTHOR_SORTS = {
    "name": "name",
    "outlet": "outlet__name",
    "articles": "article_count",
    "confidence": "article_count",
}


@staff_required
def authors(request):
    outlet_id = request.GET.get("outlet") or ""
    qs = Author.objects.all()
    if outlet_id:
        qs = qs.filter(outlet_id=outlet_id)

    key = request.GET.get("sort") or "articles"
    if key not in AUTHOR_SORTS:
        key = "articles"
    # Counts read largest-first; names read A→Z. Both still flip.
    descending = (request.GET.get("dir")
                  or ("asc" if key in ("name", "outlet") else "desc")) == "desc"
    field = AUTHOR_SORTS[key]
    order = f"-{field}" if descending else field

    return render(request, "corpus/authors.html", {
        "section": "corpus",
        **paginate(request, qs.order_by(order, "name"), 60,
                   related=("outlet",),
                   sort=key, dir="desc" if descending else "asc",
                   outlet=outlet_id),
        "outlets": Outlet.objects.all(),
        "sel_outlet": outlet_id,
        "sort": key,
        "dir": "desc" if descending else "asc",
        "flip": "asc" if descending else "desc",
        "qs_extra": query_string(outlet=outlet_id),
        "sortable": [("name", "作者"), ("outlet", "媒體"),
                     ("articles", "文章數"), ("confidence", "風格信心度")],
    })


@staff_required
def guides(request):
    return render(request, "corpus/guides.html", {
        "section": "guides",
        "guides": StyleGuide.objects.select_related("outlet", "author").all(),
        "index": EmbeddingIndex.objects.filter(name="articles").first(),
    })


@staff_required
def guide_toggle(request, pk):
    """Flip one guide's `is_active` from the list, without opening it.

    Answers 204 rather than redirecting: the switch saves itself as it is
    thrown, the same way the picture review's checkboxes do, and a redirect
    would scroll a thirty-row list back to the top on every click.

    The new state comes from the form rather than being read-and-inverted
    here — two quick clicks would otherwise race, and the second could
    reinstate what the first turned off.
    """
    guide = get_object_or_404(StyleGuide, pk=pk)
    if request.method != "POST":
        return redirect("corpus:guides")

    guide.is_active = request.POST.get("is_active") == "1"
    guide.save(update_fields=["is_active", "updated_at"])

    if request.headers.get("X-Requested-With") == "fetch":
        return HttpResponse(status=204)
    return redirect("corpus:guides")

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

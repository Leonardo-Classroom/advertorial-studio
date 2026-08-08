"""Shared paging helpers, so every list on the site pages the same way.

There used to be four hand-written pagers — one per template — each rebuilding
its own query string inline. They drifted, as copies do: the corpus list kept
its filters across pages, the run list dropped them, and none of them offered
anything but 上一頁 / 下一頁, so reaching page 40 of 67 meant clicking forty
times or editing the URL.

The template side is `templates/_pager.html`. What it needs from a view is
`page`, `page_range` (which the template cannot build — `get_elided_page_range`
takes an argument) and `pager_qs`, the rest of the query string.
"""
from __future__ import annotations

from urllib.parse import urlencode

from django.core.paginator import Paginator

# Whitelisted rather than taken from the URL: `?per=1000000` would otherwise
# be a way for any logged-in user to ask the server to render every row it has.
PAGE_SIZES = (10, 30, 50, 100)
DEFAULT_PAGE_SIZE = 10


def page_size(request, sizes=PAGE_SIZES, default: int = DEFAULT_PAGE_SIZE) -> int:
    """The page size to use, remembering the last one the user picked.

    Kept in the session rather than in every link: the choice is a preference
    about how someone likes to read these lists, so it should survive following
    a link out and coming back, which a URL-only value does not.

    A remembered value that is no longer offered falls back to the default, so
    retiring a size cannot strand whoever had it selected.
    """
    wanted = request.GET.get("per")
    if wanted:
        try:
            chosen = int(wanted)
        except ValueError:
            chosen = None
        if chosen in sizes:
            request.session["page_size"] = chosen
            return chosen

    remembered = request.session.get("page_size")
    return remembered if remembered in sizes else default


def query_string(**params) -> str:
    """Everything a paging link must carry besides `page`, ready to append.

    Empty values are dropped so an untouched filter does not litter the URL,
    and the result already starts with `&` — or is empty, when there is
    nothing to carry.
    """
    kept = {k: v for k, v in params.items() if v not in (None, "", False)}
    return ("&" + urlencode(kept)) if kept else ""


def paginate(request, queryset, per: int, related=(), **extra) -> dict:
    """Context for one paged list: the page, its elided range, and the links.

    Returned as a dict to be merged into the view's own context, so adding
    paging to a list view is one line rather than four keys to remember.

    `related` is what would otherwise be `select_related` on the queryset, and
    passing it here rather than there is the whole point. A join plus a deep
    `OFFSET` is pathological: SQLite resolves the join for every row it is
    about to skip and then throws it away, so on the 304k-row corpus list the
    last page took 36 seconds while the same offset over the bare table took
    0.19. Paging the unjoined queryset and fetching the forty rows that
    survived by primary key gives the same objects for 0.2s at any depth.
    """
    page = Paginator(queryset, per).get_page(request.GET.get("page"))
    if related and page.object_list:
        order = [obj.pk for obj in page.object_list]
        rows = queryset.model.objects.select_related(*related).in_bulk(order)
        # Rebuilt in the page's own order: `in_bulk` returns a dict, and the
        # sort this list was paged by has to survive the round trip.
        page.object_list = [rows[pk] for pk in order if pk in rows]
    return {
        "page": page,
        "page_range": page.paginator.get_elided_page_range(page.number),
        "pager_qs": query_string(**extra),
    }

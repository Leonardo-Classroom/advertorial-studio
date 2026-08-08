"""Template context every staff page needs, without every view repeating it.

Only the nav lives here. `SiteSettings.load()` is one cached row, but a
context processor runs on every render including the portal's, so this keeps
to the two flags the staff nav reads rather than exposing the whole row and
inviting templates to depend on settings they were never given deliberately.
"""
from __future__ import annotations


def nav_settings(request):
    # Anonymous and portal users never see the staff nav, so there is no reason
    # to touch the database for them.
    if not getattr(request.user, "is_staff", False):
        return {}

    from studio.models import SiteSettings

    row = SiteSettings.load()
    return {
        "show_briefs_nav": row.show_briefs_nav,
        "show_runs_nav": row.show_runs_nav,
    }

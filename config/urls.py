from django.conf import settings
from django.contrib import admin
from django.contrib.auth.decorators import login_required
from django.urls import include, path, re_path
from django.views.static import serve

urlpatterns = [
    # Portal: what an ordinary user sees. Lives at the root because it is the
    # product; the tooling below is for whoever operates it.
    path("", include("portal.urls")),

    # Everything that existed before is now staff-only, behind /manage/.
    path("manage/", include("studio.urls")),
    path("manage/corpus/", include("corpus.urls")),
    path("manage/briefs/", include("briefs.urls")),
    path("manage/users/", include("accounts.urls")),

    path("django-admin/", admin.site.urls),
]

# Uploaded deck images (`briefs/_image_review.html` renders them). Django's
# `static()` helper serves these only under DEBUG, which leaves the review page
# broken the moment DEBUG is off — and DEBUG is off whenever this is reachable
# from outside this machine. Served here instead, behind a login: a proposal
# deck's photographs are the client's unpublished material, and every other view
# in this project already requires an account.
#
# `django.views.static.serve` is single-threaded and does no caching; it is
# adequate because this runs behind `runserver` for a handful of users. Put a
# real web server in front and this route should go.
urlpatterns += [
    re_path(r"^media/(?P<path>.*)$", login_required(serve),
            {"document_root": settings.MEDIA_ROOT}),
]

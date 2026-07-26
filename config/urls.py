from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

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

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

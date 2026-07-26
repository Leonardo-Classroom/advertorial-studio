from django.contrib.auth import views as auth_views
from django.urls import path

from . import views

app_name = "portal"

urlpatterns = [
    path("", views.home, name="home"),
    path("login/", auth_views.LoginView.as_view(
        template_name="portal/login.html", redirect_authenticated_user=True,
    ), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("password/", views.change_password, name="change_password"),

    path("upload/", views.upload, name="upload"),
    path("briefs/<int:pk>/", views.brief_detail, name="brief_detail"),
    path("briefs/<int:pk>/extract/", views.brief_extract, name="brief_extract"),
    path("briefs/<int:pk>/facts/", views.brief_save_facts, name="brief_save_facts"),
    path("briefs/<int:pk>/images/", views.brief_images, name="brief_images"),
    path("briefs/<int:pk>/generate/", views.generate, name="generate"),

    path("drafts/<int:pk>/", views.draft_detail, name="draft_detail"),
    path("drafts/<int:pk>/outline/", views.draft_outline, name="draft_outline"),
    path("drafts/<int:pk>/revise/", views.draft_revise, name="draft_revise"),
    path("drafts/<int:pk>/download/", views.draft_download, name="draft_download"),
]

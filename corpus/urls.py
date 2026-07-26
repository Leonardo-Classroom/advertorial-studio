from django.urls import path

from . import views

app_name = "corpus"

urlpatterns = [
    path("", views.articles, name="articles"),
    path("articles/<int:pk>/", views.article_detail, name="article_detail"),
    path("authors/", views.authors, name="authors"),
    path("guides/", views.guides, name="guides"),
    path("guides/<int:pk>/", views.guide_detail, name="guide_detail"),
]

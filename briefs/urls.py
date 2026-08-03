from django.urls import path

from . import views

app_name = "briefs"

urlpatterns = [
    path("", views.brief_list, name="list"),
    path("upload/", views.brief_upload, name="upload"),
    path("<int:pk>/", views.brief_detail, name="detail"),
    path("<int:pk>/status/", views.brief_upload_status, name="upload_status"),
    path("<int:pk>/extract/", views.brief_extract, name="extract"),
    path("<int:pk>/facts/", views.brief_save_facts, name="save_facts"),
    path("<int:pk>/images/", views.brief_images, name="images"),
]

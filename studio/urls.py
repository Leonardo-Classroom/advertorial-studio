from django.urls import path

from . import views

app_name = "studio"

urlpatterns = [
    path("", views.home, name="home"),
    path("runs/", views.runs, name="runs"),
    path("runs/new/", views.run_new, name="run_new"),
    path("runs/<int:pk>/", views.run_detail, name="run_detail"),
    path("runs/<int:pk>/evaluate/", views.run_evaluate, name="run_evaluate"),
    path("runs/<int:pk>/revise/", views.run_revise, name="run_revise"),
    path("runs/<int:pk>/score/", views.run_score, name="run_score"),
    path("experiments/", views.experiments, name="experiments"),
    path("experiments/<int:pk>/", views.experiment_detail, name="experiment_detail"),
]

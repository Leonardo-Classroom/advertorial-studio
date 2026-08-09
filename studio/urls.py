from django.urls import path

from . import views

app_name = "studio"

urlpatterns = [
    path("", views.home, name="home"),
    path("queue/", views.queue, name="queue"),
    path("advanced/", views.advanced, name="advanced"),
    path("runs/", views.runs, name="runs"),
    path("runs/new/", views.run_new, name="run_new"),
    path("runs/<int:pk>/", views.run_detail, name="run_detail"),
    path("runs/<int:pk>/outline/", views.run_outline, name="run_outline"),
    path("runs/<int:pk>/evaluate/", views.run_evaluate, name="run_evaluate"),
    path("runs/<int:pk>/revise/", views.run_revise, name="run_revise"),
    path("runs/<int:pk>/score/", views.run_score, name="run_score"),
    path("compare/", views.compare, name="compare"),
    path("experiments/", views.experiments, name="experiments"),
    path("experiments/<int:pk>/", views.experiment_detail, name="experiment_detail"),
]

"""`runserver` with this project's default port.

Subclasses the staticfiles variant rather than the core one so static files
are still served in DEBUG — plain `django.core.management.commands.runserver`
would silently stop serving them.

An explicit address still wins: `python manage.py runserver 8000` works as
usual; only the no-argument default changes.
"""
from django.conf import settings
from django.contrib.staticfiles.management.commands.runserver import (
    Command as StaticfilesRunserverCommand,
)


class Command(StaticfilesRunserverCommand):
    default_port = str(getattr(settings, "RUNSERVER_PORT", "5860"))

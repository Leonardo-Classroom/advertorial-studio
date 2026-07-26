from django.apps import AppConfig
from django.db.backends.signals import connection_created
from django.dispatch import receiver


@receiver(connection_created)
def _sqlite_wal(sender, connection, **kwargs):
    """Put SQLite in WAL mode.

    Experiments run many generations concurrently, and the default `delete`
    journal makes a writer block readers for the length of its transaction.
    WAL lets them proceed together, which is the difference between a parallel
    experiment finishing and one thread sitting on a lock. Measured tolerance
    without it was already acceptable (8 concurrent writers, 0 failures at a
    30s busy timeout), so this is headroom rather than a fix.
    """
    if connection.vendor != "sqlite":
        return
    with connection.cursor() as cursor:
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")


class StudioConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "studio"
    verbose_name = "廣編稿生成"

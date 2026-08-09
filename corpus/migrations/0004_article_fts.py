"""Full-text index for corpus search.

`Q(title__icontains=q) | Q(body__icontains=q)` is a leading-wildcard LIKE, which
no B-tree index can serve: every one of the 300k rows gets compared, measured at
26s for titles, 28s for bodies, ~39s together (任務一 發現 5). This adds an FTS5
index so the same search is answered from the index instead.

**tokenize='trigram'**, not the default `unicode61`. The corpus is Chinese and
unicode61 does not segment it — it treats a run of Han characters as one token,
so searching 醫療 finds nothing in 智慧醫療系統. Trigram indexes every
three-character window instead, which is what makes substring search work here.
Its limit is in the name: a query shorter than three characters cannot be served
and silently matches nothing, so `corpus.views` sends those down the old LIKE
path rather than returning a confidently wrong empty page.

**content='corpus_article'** keeps this an index and not a second copy of the
corpus: FTS5 reads the columns back from the real table, and the triggers below
keep the two in step. The index still costs ~3.3GB on 220M characters of body
text — trigram indexes are large by nature, three entries per character.
"""
from django.db import migrations

CREATE = [
    # Build-time only, and per-connection: nothing here changes how the app runs
    # afterwards. SQLite's default 2MB page cache cannot hold the index b-tree
    # while it is being built, so every page split reads pages back — random
    # reads, which is the one thing this OneDrive-mounted disk is worst at. The
    # first attempt at this migration spent 30 minutes at 14% CPU, blocked in
    # uninterruptible I/O, and had written under a third of the index. With room
    # to keep the tree in memory it is CPU-bound instead.
    "PRAGMA cache_size = -8000000",   # negative = KiB, so ~7.6GiB
    "PRAGMA temp_store = MEMORY",
    """
    CREATE VIRTUAL TABLE corpus_article_fts USING fts5(
        title,
        body,
        content='corpus_article',
        content_rowid='id',
        tokenize='trigram'
    )
    """,
    # An external-content table is not populated by its own creation.
    """
    INSERT INTO corpus_article_fts(rowid, title, body)
    SELECT id, title, body FROM corpus_article
    """,
    # Deletes are recorded by re-supplying the old values, which is how FTS5
    # locates the index entries to remove — see the 'delete' command below.
    """
    CREATE TRIGGER corpus_article_fts_ai AFTER INSERT ON corpus_article BEGIN
        INSERT INTO corpus_article_fts(rowid, title, body)
        VALUES (new.id, new.title, new.body);
    END
    """,
    """
    CREATE TRIGGER corpus_article_fts_ad AFTER DELETE ON corpus_article BEGIN
        INSERT INTO corpus_article_fts(corpus_article_fts, rowid, title, body)
        VALUES ('delete', old.id, old.title, old.body);
    END
    """,
    """
    CREATE TRIGGER corpus_article_fts_au AFTER UPDATE ON corpus_article BEGIN
        INSERT INTO corpus_article_fts(corpus_article_fts, rowid, title, body)
        VALUES ('delete', old.id, old.title, old.body);
        INSERT INTO corpus_article_fts(rowid, title, body)
        VALUES (new.id, new.title, new.body);
    END
    """,
]

DROP = [
    "DROP TRIGGER IF EXISTS corpus_article_fts_au",
    "DROP TRIGGER IF EXISTS corpus_article_fts_ad",
    "DROP TRIGGER IF EXISTS corpus_article_fts_ai",
    "DROP TABLE IF EXISTS corpus_article_fts",
]


class Migration(migrations.Migration):

    dependencies = [
        ("corpus", "0003_article_corpus_article_recent_idx"),
    ]

    operations = [
        migrations.RunSQL(sql=CREATE, reverse_sql=DROP),
    ]

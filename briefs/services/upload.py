"""Create a multi-file Brief, shared by the portal and the staff backend.

Before this, the portal (`portal/views.py`) and the staff `briefs` app
(`briefs/views.py`) each had their own copy of "make a Brief, save the file,
parse it" — two implementations of the same idea that would only drift apart.
This is the one place that logic lives now; both apps call it and differ only
in how they route to it (login-gated vs `@staff_required`) and where they
redirect afterwards.
"""
from __future__ import annotations

from django.db import transaction

from briefs.models import Brief, BriefSourceFile
from briefs.services import ingest_runner, source_extract


def create_brief(owner, title: str, files) -> Brief:
    """Validate the batch, create the Brief and its source rows, queue processing.

    Format is checked up front, before anything is created: the file picker's
    `accept=".pptx"` already keeps this off the normal path, so a rejection
    here means someone bypassed it, and the whole batch is rejected rather
    than silently dropping the file they explicitly chose.

    A missing title is treated the opposite way — filled in from the first
    filename rather than rejected. The field is `required` on both upload
    forms, so an empty one arriving here also means the form was bypassed, but
    the two cases do not deserve the same answer: an unsupported format is
    something the system genuinely cannot process, whereas a nameless brief is
    merely awkward to find in a list, and a name is editable afterwards. A
    brief that reached the server with its files intact should not be thrown
    away over a label.
    """
    if not files:
        raise ValueError("請選擇至少一個檔案。")

    formats = [source_extract.format_for(f.name) for f in files]
    title = (title or "").strip()[:200] or files[0].name.rsplit(".", 1)[0][:200]

    with transaction.atomic():
        brief = Brief.objects.create(owner=owner, title=title)
        for i, (upload_file, fmt) in enumerate(zip(files, formats)):
            BriefSourceFile.objects.create(
                brief=brief, file=upload_file, format=fmt, order=i)

    ingest_runner.submit(brief)
    return brief


def status_payload(brief) -> dict:
    """Where an upload batch has got to, for the pages waiting on it."""
    files = list(brief.source_files.order_by("order", "pk"))
    ingest_runner.reap_stale(files)
    ingest_runner.reap_stale_brief(brief)
    current = next((f.original_filename for f in files if f.status == "processing"), "")
    return {
        "in_progress": brief.processing,
        "done": sum(1 for f in files if f.status == "done"),
        "total": len(files),
        "current": current,
        "failed": [f.original_filename or f.get_format_display()
                  for f in files if f.status == "failed"],
    }

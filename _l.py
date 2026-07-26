import os, django, pathlib
os.environ.setdefault("DJANGO_SETTINGS_MODULE","config.settings"); django.setup()
from briefs.models import Brief
B = pathlib.Path("/mnt/d/OneDrive/Desktop/Alan/AI人格資料for何/AI人格資料for何")
used = {pathlib.Path(b.source_file.name).name if b.source_file else None for b in Brief.objects.all()}
used_titles = [b.title for b in Brief.objects.all()]
files = sorted(p for p in B.rglob("*.pptx") if not p.name.startswith("~$"))
print(f"總共 {len(files)} 份\n")
done, todo = [], []
for p in files:
    rel = p.relative_to(B)
    # 已匯入的 4 份靠檔名關鍵字比對
    hit = any(k in p.name for k in ("0409_NB 2026 Grey Days","1201_NB 2026 Jan LNY LS",
                                    "0424_NB 2025 July FFX","0804_NB Oct Campaign"))
    (done if hit else todo).append(rel)
print(f"【已匯入 {len(done)} 份】")
for r in done: print("  ✓", r)
print(f"\n【尚未匯入 {len(todo)} 份】")
for r in todo: print("  ·", r)

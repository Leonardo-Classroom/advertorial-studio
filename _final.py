import os, django, statistics as st
os.environ.setdefault("DJANGO_SETTINGS_MODULE","config.settings"); django.setup()
from studio.models import Experiment

GROUPS = {
    "檢索策略三方比較：topical / typical / none":("跨簡報","Grey Days"),
    "檢索三方比較（Brief#2）":("跨簡報","LNY 農曆新年"),
    "檢索三方比較（Brief#3）":("跨簡報","July FFX 跑鞋"),
    "檢索三方比較（Brief#4）":("跨簡報","Oct 薄底鞋"),
    "作者級指南三方（Guide#3）":("跨作者","COOL/Arthur"),
    "作者級指南三方（Guide#4）":("跨作者","COOL/Lison"),
    "作者級指南三方（Guide#5）":("跨作者","COOL/Eddy"),
    "作者級指南三方（Guide#7）":("跨作者","GQ/Daniel Hsu"),
    "作者級指南三方（Guide#10）":("跨作者","GQ/Eric"),
}
buckets={}
print(f"{'層級':<7}{'情境':<15}{'策略':<9}{'n':>3}{'評審':>7}{'風格':>9}{'重疊':>9}{'偏離':>6}{'秒':>6}")
for e in Experiment.objects.all().order_by("pk"):
    if e.name not in GROUPS: continue
    level, label = GROUPS[e.name]
    agg={}
    for r in e.runs.filter(status="done"):
        ev=r.evaluations.first()
        if not ev or not ev.judge_dimensions: continue
        offs=sum(1 for v in (ev.deviation or {}).values() if isinstance(v,dict) and v.get("偏離")!="接近")
        row=(st.mean(dict(ev.judge_dimensions).values()), ev.style_similarity,
             ev.max_overlap, offs, r.elapsed_ms/1000, (ev.fact_coverage or {}).get("coverage"))
        agg.setdefault(r.retrieval_strategy,[]).append(row)
        buckets.setdefault((level,r.retrieval_strategy),[]).append(row)
        buckets.setdefault(("全部",r.retrieval_strategy),[]).append(row)
    for s in ("topical","typical","none"):
        v=agg.get(s,[])
        if not v: continue
        print(f"{level:<7}{label:<15}{s:<9}{len(v):>3}{st.mean(x[0] for x in v):>7.2f}"
              f"{st.mean(x[1] for x in v):>9.4f}{st.mean(x[2] for x in v):>9.4f}"
              f"{st.mean(x[3] for x in v):>6.1f}{st.mean(x[4] for x in v):>6.0f}")
print()
print(f"{'彙總':<7}{'':<15}{'策略':<9}{'n':>3}{'評審':>7}{'風格':>9}{'重疊':>9}{'偏離':>6}{'秒':>6}{'覆蓋':>7}")
for level in ("跨簡報","跨作者","全部"):
    for s in ("topical","typical","none"):
        v=buckets.get((level,s),[])
        if not v: continue
        print(f"{level:<7}{'':<15}{s:<9}{len(v):>3}{st.mean(x[0] for x in v):>7.2f}"
              f"{st.mean(x[1] for x in v):>9.4f}{st.mean(x[2] for x in v):>9.4f}"
              f"{st.mean(x[3] for x in v):>6.1f}{st.mean(x[4] for x in v):>6.0f}"
              f"{st.mean(x[5] for x in v):>7.2f}")
    print()

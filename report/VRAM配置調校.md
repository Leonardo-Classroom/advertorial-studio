# VRAM 配置調校：嵌入裝置切換 + 本地模型閒置卸載

> 2026-08-09。單張 RTX 4090（24.5GB）。針對本地後端三模型共用一張卡的
> VRAM 抖動（見任務三、任務四報告），新增兩個高級設定並實測佐證。
>
> **2026-08-09 晚間修訂**：WSL 重啟後補完了先前被 D-state 擋住的驗證。兩處原本
> 的結論被實測推翻，已改寫——見 §keep_alive 的真正生效方式、§LLM + VLM 無法共存。

---

## 問題：三個模型塞不進一張 24GB 卡

| 模型 | 跑在哪 | 權重 | **載入後實佔**（`/api/ps`）| 用途／頻率 |
|---|---|---:|---:|---|
| qwen3.6:27b-q4_K_M | Ollama | ~17 GB | **18.2 GB** | 文字，幾乎一直在用 |
| qwen3-vl:8b | Ollama | ~6.1 GB | **10.2 GB**（預設 32k context）| 視覺，只在圖片辨識時 |
| **BAAI/bge-m3** | **Django 進程（sentence-transformers）** | 2.27 GB | 2.27 GB | 嵌入，檢索時 |

**要用載入後的實佔數字，不是權重數字**——差距全在 KV cache／context，視覺模型
一項就差 4.1GB。本報告初版拿權重相加，才誤判成「騰出 2.3GB 就能共存」。

還要再扣掉 **Windows 主機端固定佔用的約 2.5 GB**（WSL 與 Windows 共用同一張卡，
`nvidia-smi` 在 WSL 內列不出主機行程，但空載時就已顯示 2.4–2.5GB）。
**Ollama 實際可用約 22 GB。**

三者無法同時常駐，Ollama 只能換入換出，付冷啟動代價
（實測冷 640s vs 暖 51s，約 12 倍）。

**關鍵**：bge-m3 不在 Ollama 裡，是本程式用 sentence-transformers 直接以 CUDA 載入
（`core/local_embeddings.py`），其 2.3GB 由本進程持有整個生命週期，**Ollama 的
`keep_alive` 管不到它**。

---

## 實測：bge-m3 GPU vs CPU

| 操作 | GPU (cuda) | CPU | 差距 |
|---|---:|---:|---:|
| 載入模型 | 6.1s | 5.9s | 幾乎相同 |
| 模型 VRAM | 2.27 GB | 0 | — |
| **查詢時嵌入**（單筆，檢索路徑）| 10 ms | **54 ms** | +44 ms |
| 語料批次嵌入（每 96 筆，建索引）| 385 ms | 4162 ms | ~11× |

**結論**：查詢多 44 毫秒——在 30–120 秒的產稿旁完全看不見。只有「重建整個語料
索引」這種一次性批次作業慢 ~11 倍（幾千筆＝幾分鐘 vs 幾十秒）。

**把 bge-m3 移到 CPU 騰出 2.27GB** 仍然值得做，但**理由不是「讓兩個模型共存」**
（見下方 §LLM + VLM 無法共存，那個推論已被實測推翻）。真正的好處是：文字模型
載入時 18.2 + 主機 2.5 = 20.7GB，只剩 3.8GB 餘裕；bge-m3 若也在卡上再吃 2.27GB
就只剩 1.5GB，逼近 Ollama 開始把層數丟回 CPU 的邊緣。移到 CPU 換來查詢多 44 毫秒，
買到的是換入換出時不會踩到部分卸載。

---

## 新增的兩個高級設定（`/manage/advanced/`）

### 1. 本地嵌入模型執行裝置（`embed_device`：cuda / cpu）

- 存於 `SiteSettings.embed_device`，預設欄位值 `cuda`（沿用 `.env` 歷史預設），
  **本機目前已切到 `cpu`**。
- `core/local_embeddings.py` 的 `get_model()` 改讀 `SiteSettings`（延遲 import，
  比照 `core/llm._backend()`，避免 core→studio 載入期依賴）；`.env` 的
  `EMBED_LOCAL_DEVICE` 仍為最終後備。
- 切換時 `studio.views.advanced` 呼叫 `local_embeddings.reset()` 丟棄已快取的模型並
  `torch.cuda.empty_cache()`，下次嵌入即以新裝置重載——不必重啟進程。
- 只在 `EMBED_BACKEND=local`（`.env`）時有意義。

### 2. 本地模型閒置卸載分鐘數（`ollama_idle_unload_minutes`，預設 3）

- 存於 `SiteSettings.ollama_idle_unload_minutes`，範圍 0–120。
- 語意：模型在最後一次請求後閒置這麼多分鐘就從 VRAM 卸載，每次請求都重設計時
  → 作業期間保持暖、閒置才釋放。這正是要的「長時間無人使用才卸載」，比
  「用完即卸」（0）好——避開每次呼叫都付 12 倍冷啟動稅。0 = 立即卸載（不建議）。
- 只影響 Ollama 的兩個模型；bge-m3 不受此控管（故另以「移到 CPU」處理）。

**生效方式（初版寫錯，已改）**：原本把 `keep_alive` 放進 `extra_body` 隨
`/v1/responses` 送出。實測 **Ollama 只有原生 `/api/*` 端點認這個欄位**，兩條
OpenAI 相容路徑都默默丟掉它：

| 送法 | 指定 | `/api/ps` 實得 |
|---|---:|---:|
| 原生 `/api/generate` | 7m（420s） | **419s** ✅ |
| `/v1/chat/completions` + `extra_body` | 9m（540s） | 419s ❌（沿用前值） |
| `/v1/responses` + `extra_body` | 11m（660s） | 419s ❌（沿用前值） |

沒有錯誤、沒有警告，只是留著伺服器預設的 5 分鐘——這個設定當時等於沒作用。

改法：`core/llm.py` 新增 `_touch_keep_alive()`，在每次**本地呼叫成功後**對
`/api/generate` 送一筆**無 prompt** 的請求（只帶 `model` 與 `keep_alive`）。
Ollama 收到無 prompt 的請求時只設計時器就回傳（`done_reason: "load"`），
實測 **0.21 秒、不重載、VRAM 不變**——擺在 30–120 秒的生成旁是雜訊。
失敗只吞掉不影響回傳值，最壞情況是這次沒改到計時器。

這樣保住了「網頁可調」；退路 `export OLLAMA_KEEP_ALIVE=3m`（伺服器級固定值）
不必動用。

---

## LLM + VLM 無法共存（初版結論錯誤，實測推翻）

初版寫「bge-m3 移 CPU 騰出 2.3GB 後，文字(17)+視覺(6.1)=23.1GB 塞得進 24.5GB，
兩者可同時常駐」。**實測是踢掉**：bge-m3 已在 CPU 的狀態下呼叫圖片辨識，
`/api/ps` 只剩視覺模型，文字模型不見了。

原因是初版拿權重數字相加。改用載入後實佔：

```
文字 18.2 + 視覺 10.2 = 28.4 GB  >  24.5 GB（更別說主機已佔 2.5GB，實際可用 ~22GB）
```

**壓 context 也救不回來**（實測 qwen3-vl:8b 在不同 `num_ctx` 的實佔）：

| num_ctx | 視覺模型 VRAM |
|---:|---:|
| 32768（預設） | 10.2 GB |
| 16384 | 7.8 GB |
| 8192 | 6.5 GB |

文字模型載入後可用空間只剩 24.5 − 18.2 − 2.5 = **3.8 GB**，而視覺模型即使壓到
8192 也要 6.5GB。再往下壓到 4096 就撞上早已記錄的矛盾：**一張圖＋提示詞實測約
4400–4500 tokens**，會直接 `exceeds the available context size` 回 400
（這正是 32B 被淘汰的原因，見 `local-vision-model-8b` 筆記）。兩個限制互斥。

**結論：這張卡上文字與視覺模型必定換入換出，無法同時常駐。** 換入換出的代價由
「閒置卸載分鐘數」控管——同一階段內（例如連續辨識 10 張圖）計時器不斷重設，
模型保持暖；換階段時付一次載入代價，而非每次呼叫都付。

---

## 建議設定（本機）

```
embed_device               = cpu   （已設定）
ollama_idle_unload_minutes = 3     （已設定，預設）
```

同一階段內（連續產稿、或連續辨識一批圖）模型保持暖；閒置 3 分鐘後自動釋放 VRAM。
文字與視覺之間的換入換出無法消除（見上節），但已收斂成「每階段一次」。

---

## keep_alive 端到端驗證：已完成 ✅

先前壓測把 Ollama server 卡進核心層 **D-state（不可中斷睡眠）**，SIGKILL 只殺掉
HTTP listener、卡住的 GPU 執行緒續佔 ~20.9GB VRAM，擋住了驗證。依
`restart_ollama.sh` 註解與 `report/本地線上API比較.md §二` 的記載重啟 WSL 後解除
（VRAM 20.9GB → 2.4GB，`/api/version` 恢復）。

驗證走真正的程式路徑（`python manage.py verify_keepalive`：改設定值 → 呼叫
`core.llm.complete()` → 讀 `/api/ps`）：

| 設定值 | 期望 | 實得 | |
|---:|---:|---:|---|
| 8 分鐘 | 480s | 479s | ✅ |
| 3 分鐘 | 180s | 179s | ✅ |

視覺路徑（`complete_vision()`）同樣驗過：呼叫後 `qwen3-vl:8b` 的計時器為 179s，
符合當時的 3 分鐘設定 ✅。

過程中發現並修掉了原本的無聲失效（`extra_body` 不被 `/v1` 端點接受，詳見上方
§生效方式）。這也是**這次驗證的主要價值**——設定看起來完全正常，程式端組出了正確
的欄位，只是對面不收。

可隨時重跑；未通過會以非零狀態結束，且無論成敗都會還原原本的設定值。

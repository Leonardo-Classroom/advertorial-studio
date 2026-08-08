#!/usr/bin/env bash
# 給校內區網（如 192.168.52.212）使用，對應 .env 的 DJANGO_DEBUG=0
set -e

cd "$(dirname "$0")"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate leo3.10
fi

# 本地 LLM 後端（SiteSettings.llm_backend == "local"）走 Ollama，沒開就會讓
# 圖片辨識、文字產出連線失敗。這裡只確保它「有在跑」，不強制重啟——已經
# 活著的話 _ollama_start 什麼都不做（見 restart_ollama.sh 的說明：重啟要付
# 模型冷啟動代價，不是免費的）。啟動失敗不擋這支腳本本身：只用線上後端
# 的話本來就不需要它。
source ./restart_ollama.sh
_ollama_start || echo "Ollama 未啟動——若目前用本地後端，功能會連線失敗。可執行 ./restart_ollama.sh 排查。"

python manage.py runserver 0.0.0.0:5860 --insecure

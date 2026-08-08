#!/usr/bin/env bash
# 本地 Ollama server 的共用控制邏輯：確認活著、啟動、停止。
#
# 直接執行本腳本（`./restart_ollama.sh`）會做「先停舊的、再啟新的、等到
# 真的能連上為止」的完整重啟。start.sh 改成用 `source` 引入這裡的函式，
# 只在 Ollama 還沒起來時呼叫 `_ollama_start`，不會每次開伺服器都白白重開
# 一次模型——文字模型跟視覺模型共用一張 24GB 顯卡、裝不下同時常駐，
# 換入換出要付冷啟動代價（見 report/本地線上API比較.md §二），沒事不重啟。
#
# 已知過一次事故（同上文件 §二）：Ollama server 卡進核心層級的 D state
# （disk sleep，不可中斷睡眠），連 kill -9 都無效——那是核心排程器在等一個
# I/O 完成，不是可以用訊號處理的狀態。遇到殺不掉時這裡會直接說明並放棄
# 重試，而不是假裝還有救；真正的解法只有重開 WSL，或换一個 port 啟動新的
# server（`OLLAMA_PORT=11435 ./restart_ollama.sh`，同一份文件記的繞過法，
# 不去搶被卡住的舊 port）。

_ollama_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 位址／埠號預設跟著 .env 的 LOCAL_LLM_BASE_URL 走，不另外寫死一份。這兩邊
# 曾經各說各話過一次，代價不小：.env 指著 11435（一個早就沒在跑的繞過用
# instance），Ollama 卻起在 11434，於是每次「解析勾選的圖片」都在幾秒內
# 吃到 APIConnectionError、GPU 完全沒動，畫面上卻只顯示「已辨識 N 張，
# 其中 0 張判定可用」——失敗被降級成 unknown，看起來就只像模型判不出來。
# 讓腳本讀同一行設定，這種不一致就不可能再發生。
_ollama_env_url="$(sed -n 's/^[[:space:]]*LOCAL_LLM_BASE_URL[[:space:]]*=[[:space:]]*//p' \
  "$_ollama_dir/.env" 2>/dev/null | tail -1 | tr -d '"'"'"' \r')"
_ollama_env_hostport="${_ollama_env_url#*://}"   # 去掉 scheme
_ollama_env_hostport="${_ollama_env_hostport%%/*}"  # 去掉 /v1 之類的路徑

OLLAMA_HOST_ADDR="${OLLAMA_HOST_ADDR:-${_ollama_env_hostport%%:*}}"
OLLAMA_HOST_ADDR="${OLLAMA_HOST_ADDR:-127.0.0.1}"
OLLAMA_PORT="${OLLAMA_PORT:-${_ollama_env_hostport##*:}}"
# `##*:` 在沒有冒號時會原封不動吐回主機名，那不是埠號——只認純數字。
[[ "$OLLAMA_PORT" =~ ^[0-9]+$ ]] || OLLAMA_PORT=11434
OLLAMA_LOG="$_ollama_dir/var/ollama.log"
OLLAMA_PIDFILE="$_ollama_dir/var/ollama.pid"

_ollama_is_up() {
  curl -s -m 2 "http://$OLLAMA_HOST_ADDR:$OLLAMA_PORT/api/version" >/dev/null 2>&1
}

# 等最多 $1 秒（預設 30）讓 server 回應。
_ollama_wait_ready() {
  local timeout="${1:-30}"
  for ((i = 0; i < timeout; i++)); do
    _ollama_is_up && return 0
    sleep 1
  done
  return 1
}

_ollama_start() {
  if _ollama_is_up; then
    echo "Ollama 已在 $OLLAMA_HOST_ADDR:$OLLAMA_PORT 回應，不重複啟動。"
    return 0
  fi
  if ! command -v ollama >/dev/null 2>&1; then
    echo "找不到 ollama 指令，略過（若只用線上後端可忽略）。" >&2
    return 1
  fi
  mkdir -p "$_ollama_dir/var"
  echo "啟動 Ollama server（log：$OLLAMA_LOG）…"
  OLLAMA_HOST="$OLLAMA_HOST_ADDR:$OLLAMA_PORT" nohup ollama serve >>"$OLLAMA_LOG" 2>&1 &
  echo $! > "$OLLAMA_PIDFILE"
  if _ollama_wait_ready 30; then
    echo "Ollama 已就緒：http://$OLLAMA_HOST_ADDR:$OLLAMA_PORT"
    return 0
  fi
  echo "等了 30 秒仍未就緒，請檢查 $OLLAMA_LOG。" >&2
  return 1
}

_ollama_stop() {
  # 刻意不用 _ollama_is_up 當作「有沒有在跑」的判斷。卡死的 server 正是
  # 不回應 /api/version、卻仍然佔著 11434 的那種狀態——用回應與否判斷會直接
  # 跳過停止、接著啟動就撞 "bind: address already in use"（實際發生過）。
  # 這裡改看「有沒有 process」，那才是能不能啟動新的的真正條件。
  local pids
  pids="$(_ollama_pids)"
  [ -z "$pids" ] && return 0

  kill $pids 2>/dev/null || true
  if _ollama_wait_gone 10; then
    rm -f "$OLLAMA_PIDFILE"
    return 0
  fi

  echo "SIGTERM 沒能讓 server 停下，改用 SIGKILL。"
  kill -9 $pids 2>/dev/null || true
  if _ollama_wait_gone 5; then
    rm -f "$OLLAMA_PIDFILE"
    return 0
  fi

  echo "無法停止 $OLLAMA_HOST_ADDR:$OLLAMA_PORT 上的 Ollama（pid：$(_ollama_pids | tr '\n' ' ')）" \
       "——process 卡進核心 D state（不可中斷睡眠）時，kill -9 也無效，因為那是核心" \
       "排程器層級的狀態，不會回應任何訊號。見 report/本地線上API比較.md §二。"
  echo "已知解法有兩個：(1) 重開 WSL；(2) 換一個埠口起新的 instance，不去搶卡住的" \
       "$OLLAMA_PORT — 但別忘了 .env 的 LOCAL_LLM_BASE_URL 要一起改，否則 app 仍" \
       "連向舊埠口（這個不一致踩過一次）。"
  return 1
}

# 這個 server 的 pid。優先用啟動時記下的，避免 pgrep 誤殺到不是這裡起的 ollama；
# 記錄遺失（例如手動啟動過）時才退回比對「聽在這個埠口上的那支」。
_ollama_pids() {
  if [ -f "$OLLAMA_PIDFILE" ] && kill -0 "$(cat "$OLLAMA_PIDFILE")" 2>/dev/null; then
    cat "$OLLAMA_PIDFILE"
    return
  fi
  # 用埠口反查比 pgrep 準：真正擋住新 server 的就是佔著這個埠口的那支。
  #
  # 每個查詢都包 timeout：`ss -p` 和 `pgrep` 都要讀 /proc，而讀一個卡在
  # D state 的 process 的 /proc 自己也會卡住（實際發生過，連 ps 都掛）。
  # 排查工具被它要排查的對象卡死，是最糟的失敗方式——寧可回報不出 pid，
  # 也不要讓整個腳本停在這裡。
  local owner
  owner="$(timeout 5 ss -lntp 2>/dev/null | awk -v p=":$OLLAMA_PORT\$" '$4 ~ p' \
           | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u)"
  if [ -n "$owner" ]; then
    echo "$owner"
    return
  fi
  timeout 5 pgrep -f "ollama serve" || true
}

# 等到 process 真的不見為止。刻意不是等「server 沒回應」——卡死的 server 早就
# 不回應了，卻仍佔著埠口，用回應與否判斷會誤判成已經停掉。
_ollama_wait_gone() {
  local timeout="${1:-10}" i
  for ((i = 0; i < timeout; i++)); do
    [ -z "$(_ollama_pids)" ] && return 0
    sleep 1
  done
  [ -z "$(_ollama_pids)" ]
}

# 只有直接執行本檔（`./restart_ollama.sh`）才跑完整的停止＋啟動；被
# start.sh 用 source 引入時，只是借用上面的函式，不會自動重啟。
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "重啟本地 Ollama server…"
  _ollama_stop || exit 1
  _ollama_start || exit 1
fi

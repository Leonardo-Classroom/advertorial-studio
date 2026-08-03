#!/usr/bin/env bash
# 給校內區網（如 192.168.52.212）使用，對應 .env 的 DJANGO_DEBUG=0
set -e

cd "$(dirname "$0")"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate leo3.10
fi

python manage.py runserver 0.0.0.0:5860 --insecure

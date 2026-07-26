#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "無法匯入 Django。請確認已啟用 conda 環境 leo3.10 並安裝 requirements.txt。"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()

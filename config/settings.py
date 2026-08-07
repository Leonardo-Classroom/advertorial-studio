"""Django settings for 廣編系統 (advertorial generation studio).

Config lives in `.env` at the project root so the backing model / corpus path
can be swapped without touching code. Parsed by hand to avoid depending on
python-dotenv.
"""
from __future__ import annotations

import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    pairs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if m:
            pairs[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return pairs


ENV = _load_env(BASE_DIR / ".env")


def env(key: str, default: str = "") -> str:
    import os
    return ENV.get(key) or os.environ.get(key) or default


def env_int(key: str, default: int) -> int:
    try:
        return int(env(key) or default)
    except ValueError:
        return default


def env_bool(key: str, default: bool = False) -> bool:
    raw = env(key)
    if raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


SECRET_KEY = env("DJANGO_SECRET_KEY", "dev-insecure-key-change-me")
DEBUG = env_bool("DJANGO_DEBUG", True)
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    # `studio` must come before django.contrib.staticfiles: Django resolves a
    # management command to the earliest app in this list that defines it, and
    # studio overrides `runserver` to default to RUNSERVER_PORT.
    "studio",
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corpus",
    "briefs",
    "accounts",
    "portal",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
        "OPTIONS": {"timeout": 30},
    }
}

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = "zh-hant"
TIME_ZONE = "Asia/Taipei"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"] if (BASE_DIR / "static").exists() else []
MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Two audiences: the portal at / for people who just want a draft, and the
# tooling under /manage/ for whoever runs the system. Anonymous visitors land
# on the portal login, not the Django admin one.
LOGIN_URL = "portal:login"
LOGIN_REDIRECT_URL = "portal:home"
LOGOUT_REDIRECT_URL = "portal:login"

# Default port for `manage.py runserver` (see studio/management/commands/runserver.py).
RUNSERVER_PORT = env("RUNSERVER_PORT", "5860")

# --- Application-specific config -------------------------------------------

# Generation model (Azure AI Services, OpenAI-compatible Responses API).
LLM_BASE_URL = env("LLM_BASE_URL", "https://leo-test-0627-ai.services.ai.azure.com/openai/v1")
LLM_API_KEY = env("LLM_API_KEY")
LLM_MODEL = env("LLM_MODEL", "gpt-5.4")
# The judge is pinned separately. Comparing two writing models while the marker
# changes with them measures nothing — each model would be graded by itself.
LLM_JUDGE_MODEL = env("LLM_JUDGE_MODEL", "") or LLM_MODEL
LLM_TIMEOUT = env_int("LLM_TIMEOUT", 180)
LLM_SEND_TEMPERATURE = env_bool("LLM_SEND_TEMPERATURE", False)
LLM_TEMPERATURE = float(env("LLM_TEMPERATURE", "0.8") or 0.8)

# Local completion backend (Ollama), selected at runtime via SiteSettings.
# llm_backend — same "which provider" / "which model" split as EMBED_BACKEND
# below: whether to use it lives in the DB (an operational switch staff can
# flip without a deploy), which model/endpoint it points at lives here.
#
# Two real, measured costs of flipping this on — not hypothetical, see
# report/本地線上API比較.md: text generation has ~20% chance of running
# 5-50x longer than normal (Qwen3.6 occasionally not fully suppressing its
# thinking mode despite `reasoning:{"effort":"none"}`, cause unconfirmed);
# image classification has ~30% chance of failing outright or returning a
# blank description. `LOCAL_LLM_TIMEOUT` is generous on purpose — round 2's
# benchmark forgot to wire a timeout through at all and one call hung 30
# minutes; failing at 300s beats that, but is still a long wait for a user.
LOCAL_LLM_BASE_URL = env("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
LOCAL_LLM_API_KEY = env("LOCAL_LLM_API_KEY", "ollama")
LOCAL_LLM_MODEL = env("LOCAL_LLM_MODEL", "qwen3.6:27b-q4_K_M")
LOCAL_LLM_VISION_MODEL = env("LOCAL_LLM_VISION_MODEL", "qwen3-vl:32b-fast")
LOCAL_LLM_TIMEOUT = env_int("LOCAL_LLM_TIMEOUT", 300)

# Embedding model. This Azure resource has no text-embedding-3-* deployment;
# Cohere Embed v4 is what is actually reachable (verified 2026-07-26).
EMBED_MODEL = env("EMBED_MODEL", "embed-v-4-0")
EMBED_DIMENSIONS = env_int("EMBED_DIMENSIONS", 1024)
EMBED_BATCH_SIZE = min(env_int("EMBED_BATCH_SIZE", 96), 96)  # hard API limit
EMBED_CONCURRENCY = env_int("EMBED_CONCURRENCY", 4)
# Free-tier deployments cap requests per minute (429 RateLimitReached).
# Because the cap counts requests rather than tokens, indexing throughput is
# governed by how full each request is, not by concurrency.
EMBED_RPM = env_int("EMBED_RPM", 10)
# Characters of each article fed to the embedder. Shorter means more articles
# fit per request, which is what actually speeds up indexing under an RPM cap.
EMBED_INDEX_CHARS = env_int("EMBED_INDEX_CHARS", 350)

# Embedding backend: "local" runs a model on this machine's GPU, "api" calls
# the Azure/Cohere deployment. Local is the default because the remote free
# tier allows only 50 requests/day (~850 articles), which makes indexing the
# corpus take weeks.
EMBED_BACKEND = env("EMBED_BACKEND", "local")
EMBED_LOCAL_MODEL = env("EMBED_LOCAL_MODEL", "BAAI/bge-m3")
EMBED_LOCAL_DEVICE = env("EMBED_LOCAL_DEVICE", "cuda")
EMBED_LOCAL_BATCH = env_int("EMBED_LOCAL_BATCH", 64)

# Where the crawled style corpus lives: <outlet>/<author>/*.txt beneath this.
# Inside the project (and git-ignored) rather than at an absolute path outside
# it, so a checkout on another machine only has to drop the corpus in place.
CORPUS_ROOT = Path(env("CORPUS_ROOT", str(BASE_DIR / "datasets" / "news")))

# Vector index files (numpy matrices + id maps) live here.
INDEX_DIR = BASE_DIR / "var" / "index"

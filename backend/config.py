import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "") or SUPABASE_SERVICE_KEY

FRONTEND_ORIGINS = [
    o.strip()
    for o in os.getenv("FRONTEND_ORIGINS", "http://localhost:8501").split(",")
    if o.strip()
]
ADMIN_EMAILS = {
    e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()
}

MAX_PDFS_PER_USER = _int("MAX_PDFS_PER_USER", 3)
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 25)
MAX_FILES_PER_REQUEST = _int("MAX_FILES_PER_REQUEST", 10)
MAX_CONCURRENT_INGEST = _int("MAX_CONCURRENT_INGEST", 2)
MAX_CONCURRENT_HEAVY = _int("MAX_CONCURRENT_HEAVY", 1)
QUEUE_TIMEOUT_SECONDS = _int("QUEUE_TIMEOUT_SECONDS", 30)
TOKEN_CACHE_SECONDS = _int("TOKEN_CACHE_SECONDS", 60)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".html", ".htm", ".csv", ".xlsx"}

REPORT_DIR = ROOT / "evaluation" / "reports"
CLASSIFIER_TEST_CSV = ROOT / "classifier" / "data" / "test.csv"

import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_tmp = tempfile.mkdtemp()
os.environ.update(UPLOAD_DIR=f"{_tmp}/up", INDEX_DIR=f"{_tmp}/idx", OUTPUT_DIR=f"{_tmp}/out",
                  DATABASE_URL=f"sqlite:///{_tmp}/t.db", EMBEDDING_MODEL="none", APP_API_KEY="")

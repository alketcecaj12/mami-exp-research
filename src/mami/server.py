"""Optional local HTTP server; run the model on macOS, not inside Linux Docker."""
from threading import Lock
from contextlib import asynccontextmanager
import base64
import tempfile
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .config import load_config
from .vlm import Classifier

lock = Lock()


@asynccontextmanager
async def lifespan(app):
    app.state.classifier = Classifier(load_config("configs/default.yaml"))
    yield


app = FastAPI(lifespan=lifespan)


class Request(BaseModel):
    image_base64: str
    text: str = ""


@app.get("/health")
def health():
    return {"status": "ready"}


@app.post("/classify")
def classify(request: Request):
    try:
        image_bytes = base64.b64decode(request.image_base64, validate=True)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid base64 image")
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "image"
        path.write_bytes(image_bytes)
        with lock:
            return app.state.classifier.predict(path, request.text)

from fastapi import FastAPI, Form, UploadFile, File
from fastapi import HTTPException, status

import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Union, Optional

import numpy as np
import whisper

app = FastAPI()

#url https://api.openai.com/v1/audio/transcriptions \
#  -H "Authorization: Bearer $OPENAI_API_KEY" \
#  -H "Content-Type: multipart/form-data" \
#  -F model="whisper-1" \
#  -F file="@/path/to/file/openai.mp3"

#{
#  "text": "Imagine the wildest idea that you've ever had, and you're curious about how it might scale to something that's a 100, a 1,000 times bigger..."
#}

# -----
# copied from https://github.com/hayabhay/whisper-ui

# Whisper transcription functions
# ----------------
@lru_cache(maxsize=1)
def get_whisper_model(whisper_model: str):
    """Get a whisper model from the cache or download it if it doesn't exist"""
    model = whisper.load_model(whisper_model)
    return model

def transcribe(audio_path: str, whisper_model: str, **whisper_args):
    """Transcribe the audio file using whisper"""

    # Get whisper model
    # NOTE: If mulitple models are selected, this may keep all of them in memory depending on the cache size
    transcriber = get_whisper_model(whisper_model)

    # Set configs & transcribe
    if whisper_args["temperature_increment_on_fallback"] is not None:
        whisper_args["temperature"] = tuple(
            np.arange(whisper_args["temperature"], 1.0 + 1e-6, whisper_args["temperature_increment_on_fallback"])
        )
    else:
        whisper_args["temperature"] = [whisper_args["temperature"]]

    del whisper_args["temperature_increment_on_fallback"]

    transcript = transcriber.transcribe(
        audio_path,
        **whisper_args,
    )

    return transcript


WHISPER_DEFAULT_SETTINGS = {
    "whisper_model": "base",
#    "whisper_model": "large-v2",
    "temperature": 0.0,
    "temperature_increment_on_fallback": 0.2,
    "no_speech_threshold": 0.6,
    "logprob_threshold": -1.0,
    "compression_ratio_threshold": 2.4,
    "condition_on_previous_text": True,
    "verbose": False,
    "task": "transcribe",
#    "task": "translation",
}

UPLOAD_DIR="/tmp"
# -----

@app.post('/v1/audio/transcriptions')
async def transcriptions(model: str = Form(...), file: UploadFile = File(...), response_format: Optional[str] = Form(None)):

    assert model == "whisper-1"
    if file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Bad Request"
            )

    filename = file.filename
    fileobj = file.file
    upload_name = os.path.join(UPLOAD_DIR, filename)
    upload_file = open(upload_name, 'wb+')
    shutil.copyfileobj(fileobj, upload_file)
    upload_file.close()

    transcript = transcribe(audio_path=upload_name, **WHISPER_DEFAULT_SETTINGS)

    if response_format == 'text':
        return transcript['text']

    return {"text": transcript['text']}
    return {transcript}


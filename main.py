from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, Form, UploadFile, File
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse

import os
import shutil
from pathlib import Path
from typing import Any, List, Union, Optional

from datetime import timedelta

import numpy as np
import whisper
import torch

import uvicorn
import json
import base64
import tempfile


#url https://api.openai.com/v1/audio/transcriptions \
#  -H "Authorization: Bearer $OPENAI_API_KEY" \
#  -H "Content-Type: multipart/form-data" \
#  -F model="whisper-1" \
#  -F file="@/path/to/file/openai.mp3"

#{
#  "text": "Imagine the wildest idea that you've ever had, and you're curious about how it might scale to something that's a 100, a 1,000 times bigger..."
#}

WHISPER_MODEL = os.environ.get('WHISPER_MODEL', 'turbo')
CHAT_MODEL = os.environ.get('CHAT_MODEL', None)

whisper_model = None
chat_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_model
    # Load the ML model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = whisper.load_model(WHISPER_MODEL, device=device, in_memory=True)

    yield

    # Clean up the ML models and release the resources
    del whisper_model
    whisper_model = None

app = FastAPI(lifespan=lifespan)

# -----
# copied from https://github.com/hayabhay/whisper-ui
# Whisper transcription functions
def transcribe(audio_path: str, **whisper_args):
    """Transcribe the audio file using whisper"""
    global whisper_model

    # Set configs & transcribe
    if whisper_args["temperature_increment_on_fallback"] is not None:
        whisper_args["temperature"] = tuple(
            np.arange(whisper_args["temperature"], 1.0 + 1e-6, whisper_args["temperature_increment_on_fallback"])
        )
    else:
        whisper_args["temperature"] = [whisper_args["temperature"]]

    del whisper_args["temperature_increment_on_fallback"]

    transcript = whisper_model.transcribe(
        audio_path,
        **whisper_args,
    )

    return transcript

WHISPER_DEFAULT_SETTINGS = {
#    "whisper_model": "turbo",
    "temperature": 0.0,
    "temperature_increment_on_fallback": 0.2,
    "no_speech_threshold": 0.6,
    "logprob_threshold": -1.0,
    "compression_ratio_threshold": 2.4,
    "condition_on_previous_text": True,
    "verbose": False,
    "task": "transcribe",
    
    #
#    "initial_prompt": None,
#    "carry_initial_prompt": False,
#    "word_timestamps": False,
#    "prepend_punctuations": "\"'“¿([{-",
#    "append_punctuations": "\"'.。,，!！?？:：”)]}、",
#    "clip_timestamps": "0",
#    "hallucination_silence_threshold": None,
#    "verbose": None,
}

import tempfile
UPLOAD_DIR=tempfile.gettempdir()
#UPLOAD_DIR="/tmp"


@app.get('/v1/models')
async def v1_models(request: Request):
    content = {
        "object": "list",
        "data": [
            {
                "id": "whisper-1",
                "object": "model",
                "created": 17078881749,
                "owned_by": "tiny-whisper-api"
            },
            {
                "id": "gpt-4o-audio-preview",
                "object": "model",
                "created": 17078881749,
                "owned_by": "tiny-whisper-api"
            },
            {
                "id": "gpt-4o-audio-preview-2024-10-01",
                "object": "model",
                "created": 17078881749,
                "owned_by": "tiny-whisper-api"
            }
        ]
    }

    headers = {
        'Content-Type': 'application/json'
    }

    response_status_code = 200

    resp = JSONResponse(
        content = content,
        headers = headers,
        status_code = response_status_code
    )

    return resp

# gpt-4o-audio-preview：OpenAI の Chat Completions API でオーディオを扱う新機能を軽く見てみる【2024/10/17リリース】 #ChatGPT - Qiita
# https://qiita.com/youtoy/items/5a87fd22cc88d8c34d6d

# https://platform.openai.com/docs/api-reference/chat/create
'''
curl "https://api.openai.com/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -d '{
      "model": "gpt-4o-audio-preview",
      "modalities": ["text", "audio"],
      "audio": { "voice": "alloy", "format": "wav" },
      "messages": [
        {
          "role": "user",
          "content": [
            { "type": "text", "text": "What is in this recording?" },
            {
              "type": "input_audio",
              "input_audio": {
                "data": "<base64 bytes here>",
                "format": "wav"
              }
            }
          ]
        }
      ]
    }'
'''

# modalities = ["text"]
CHAT_COMPLETIONS_RESPONSE_TEMPLATE='''
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-4o-audio-preview-2024-10-01",
  "system_fingerprint": "fp_44709d6fcb",
  "service_tier": null,
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello there, how may I assist you today?",
      "refusal": null
    },
    "logprobs": null,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 86,
    "prompt_tokens_details": {
      "audio_tokens": 69,
      "cached_tokens": 0,
      "text_tokens": 17,
      "image_tokens": 0
    },
    "completion_tokens": 36,
    "total_tokens": 122,
    "completion_tokens_details": {
      "reasoning_tokens": 0,
      "accepted_prediction_tokens": 0,
      "rejected_prediction_tokens": 0,
      "audio_tokens": 0,
      "reasoning_tokens": 0,
      "text_toekns": 17
    }
  }
}
'''

# modalities = ["text", "audio"]
CHAT_COMPLETIONS_RESPONSE_AUDIO_OUTPUT_TEMPLATE='''
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-4o-audio-preview-2024-10-01",
  "system_fingerprint": "fp_44709d6fcb",
  "service_tier": null,
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "refusal": null,
      "audio": {
        "id": "audio_6744555cc6d48190b67e70798ab606c3",
        "data": "response_audio_data_base64",
        "expires_at": 1732535148,
        "transcript": "response_transcript"
      }
    },
    "logprobs": null,
    "finish_reason": "stop",
    "function_call": null,
    "tool_calls": null
  }],
  "usage": {
    "prompt_tokens": 86,
    "prompt_tokens_details": {
      "audio_tokens": 69,
      "cached_tokens": 0,
      "text_tokens": 17,
      "image_tokens": 0
    },
    "completion_tokens": 236,
    "total_tokens": 322,
    "completion_tokens_details": {
      "reasoning_tokens": 0,
      "accepted_prediction_tokens": 0,
      "rejected_prediction_tokens": 0,
      "audio_tokens": 188,
      "reasoning_tokens": 0,
      "text_tokens": 48
    }
  }
}
'''

CHAT_COMPLETIONS_RESPONSE_DIFY_PING_TEMPLATE='''
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-4o-audio-preview-2024-10-01",
  "system_fingerprint": "fp_44709d6fcb",
  "service_tier": null,
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Pong",
      "refusal": null
    },
    "logprobs": null,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 5,
    "prompt_tokens_details": {
      "audio_tokens": 0,
      "cached_tokens": 0,
      "text_tokens": 5,
      "image_tokens": 0
    },
    "completion_tokens": 5,
    "total_tokens": 10,
    "completion_tokens_details": {
      "reasoning_tokens": 0,
      "accepted_prediction_tokens": 0,
      "rejected_prediction_tokens": 0,
      "audio_tokens": 0,
      "reasoning_tokens": 0,
      "text_toekns": 5
    }
  }
}
'''

def is_base64_encoded(s: str) -> bool:
    try:
        # パディングの調整（4の倍数に）
        if len(s) % 4 != 0:
            return False

        # デコードしてみる
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False

def save_base64_to_temp_file(base64_string: str) -> str:
    try:
        # BASE64文字列をデコード
        binary_data = base64.b64decode(base64_string)

        # 一時ファイルを作成
        with tempfile.NamedTemporaryFile(delete=False, mode='wb') as temp_file:
            temp_file.write(binary_data)
            temp_file_path = temp_file.name  # 一時ファイルのパスを取得
        
        return temp_file_path

    except Exception as e:
        return None

@app.post('/v1/chat/completions')
async def v1_chat_completions(request: Request):

    global chat_model

    req_body = await request.json()

    model = req_body['model']
    try:
        modalities = req_body['modalities']
    except KeyError:
        modalities = ['text']
    try:
        audio = req_body['audio']
    except KeyError:
        audio = None

    if model not in ['gpt-4o-audio-preview', 'gpt-4o-audio-preview-2024-10-01']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Bad Request, not supported model"
            )

    if 'text' not in modalities:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Bad Request, 'text' is not in modalitiees"
            )
    
    if 'audio' in modalities:
        if audio is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Bad Request, 'audio' is in modalitiees, but attiributes are not specified."
                )

        if audio['voice'] not in ['ash', 'ballad', 'coral', 'sage', 'verse', 'alloy', 'echo', 'shmmer']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Bad Request, not supported voice"
                )
        if audio['format'] not in ['wav', 'mp3', 'flac', 'opus', 'pcm16']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Bad Request, not supported format"
                )

    messages = req_body['messages']
    content = None
    for m in messages:
        for c in m['content']:
            if 'input_audio' in c:
                assert 'data' in c['input_audio']
                assert 'format' in c['input_audio']
                content = c
                break

    if content is None:
        # dify ping
        for m in messages:
            if m['content'] in ['ping']:
                resp_body = json.loads(CHAT_COMPLETIONS_RESPONSE_DIFY_PING_TEMPLATE)
                resp_body['model'] = model
                resp = JSONResponse(
                    content = resp_body,
                    headers = {
                        'Content-Type': 'application/json'
                    },
                    status_code = 200
                )
                return resp
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Bad Request, missing content"
            )

    # content data is base64-encoded?
    data = content['input_audio']['data']
    if not is_base64_encoded(data):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Bad Request, content is not base64-encoded."
            )

    settings = WHISPER_DEFAULT_SETTINGS.copy()
    #settings['temperature'] = temperature

    temp_content_path = save_base64_to_temp_file(data)
    if temp_content_path is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Bad Request, transcription failed."
            )

    transcript = transcribe(audio_path=temp_content_path, **settings)
    text = transcript['text']

    if temp_content_path:
        # TODO: 非同期で削除したい
        os.remove(temp_content_path)

    #print(transcript)
    #print(text)

    if audio is not None:
        resp_body = json.loads(CHAT_COMPLETIONS_RESPONSE_AUDIO_OUTPUT_TEMPLATE)
        resp_body['choices'][0]['message']['audio']['transcript'] = text
    else:
        resp_body = json.loads(CHAT_COMPLETIONS_RESPONSE_TEMPLATE)
        resp_body['choices'][0]['message']['content'] = text
        resp_body['choices'][0]['delta'] = resp_body['choices'][0]['message'].copy()

    resp_body['model'] = model

    if chat_model:
        pass
        # request の messagesのうち、 input_audio 部分を書き起こしに差し替えて、
        # chat_model に投げつけて応答を得る。
        # ? chat_modelのAPI_KEYは? 接続先は?
        # ? トークン数などの統計情報をどう補正する?
        # ? modalities 情報は? 音声出力しないのであれば気にしなくてよい?

    resp = JSONResponse(
        content = resp_body,
        headers = {
            'Content-Type': 'application/json'
        },
        status_code = 200
    )
    
    return resp


@app.post('/v1/audio/transcriptions')
async def transcriptions(model: str = Form(...),
                         file: UploadFile = File(...),
                         response_format: Optional[str] = Form(None),
                         language: Optional[str] = Form(None),
                         prompt: Optional[str] = Form(None),
                         temperature: Optional[float] = Form(None)):

    assert model == "whisper-1"
    if file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Bad Request, bad file"
            )
    if response_format is None:
        response_format = 'json'
    if response_format not in ['json',
                           'text',
                           'srt',
                           'verbose_json',
                           'vtt']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Bad Request, bad response_format"
            )
    if temperature is None:
        temperature = 0.0
    if temperature < 0.0 or temperature > 1.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Bad Request, bad temperature"
            )

    filename = file.filename
    fileobj = file.file
    
    upload_name = None
    try:
        # TODO: 拡張子は維持しながら、ファイル名の衝突を避けるために一時ファイルとしたい
        upload_name = os.path.join(UPLOAD_DIR, filename)
        upload_file = open(upload_name, 'wb')
        shutil.copyfileobj(fileobj, upload_file)
        upload_file.close()

        # 指定されたパラメータで上書き
        settings = WHISPER_DEFAULT_SETTINGS.copy()
        settings['temperature'] = temperature
        if language is not None:
            settings['language'] = language # TODO: check  ISO-639-1  format

        # -F prompt="The following conversation is a lecture about the recent developments around OpenAI, GPT-4.5 and the future of AI."
        if prompt is not None:
            settings['initial_prompt'] = prompt
            # プロンプトを各音声セグメントに適用するか、先頭セグメントのみ適用(デフォルト)か。
            carry_initial_prompt = None
            if carry_initial_prompt is not None:
                settings['carry_initial_prompt'] = carry_initial_prompt

        # -F "timestamp_granularities[]=word"
        timestamp_granularities = None
        if timestamp_granularities is not None:
            if timestamp_granularities == 'word':
                settings['word_timestamps'] = True

        # -F stream=True  # whisper-1 not supported
        stream = None
        if stream is not None:
            pass    # just drop. whisper-1 not support streaming output

        transcript = transcribe(audio_path=upload_name, **settings)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error: transcription failed."
            )
    finally:
        if upload_name:
            os.remove(upload_name)


    if response_format in ['text']:
        return Response(content=transcript['text'], media_type="text/plain")

    if response_format in ['srt']:
        ret = ""
        for seg in transcript['segments']:
            
            td_s = timedelta(milliseconds=seg["start"]*1000)
            td_e = timedelta(milliseconds=seg["end"]*1000)

            t_s = f'{td_s.seconds//3600:02}:{(td_s.seconds//60)%60:02}:{td_s.seconds%60:02},{td_s.microseconds//1000:03}'
            t_e = f'{td_e.seconds//3600:02}:{(td_e.seconds//60)%60:02}:{td_e.seconds%60:02},{td_e.microseconds//1000:03}'

            ret += '{}\n{} --> {}\n{}\n\n'.format(seg["id"], t_s, t_e, seg["text"].strip())
        ret += '\n'
        return Response(content=ret, media_type="text/plain")

    if response_format in ['vtt']:
        ret = "WEBVTT\n\n"
        for seg in transcript['segments']:
            td_s = timedelta(milliseconds=seg["start"]*1000)
            td_e = timedelta(milliseconds=seg["end"]*1000)

            t_s = f'{td_s.seconds//3600:02}:{(td_s.seconds//60)%60:02}:{td_s.seconds%60:02}.{td_s.microseconds//1000:03}'
            t_e = f'{td_e.seconds//3600:02}:{(td_e.seconds//60)%60:02}:{td_e.seconds%60:02}.{td_e.microseconds//1000:03}'

            ret += "{} --> {}\n{}\n\n".format(t_s, t_e, seg["text"].strip())
        return Response(content=ret, media_type="text/plain")

    if response_format in ['verbose_json']:
        transcript.setdefault('task', WHISPER_DEFAULT_SETTINGS['task'])
        transcript.setdefault('duration', transcript['segments'][-1]['end'])
        if transcript['language'] == 'ja':
            transcript['language'] = 'japanese'
        return transcript

    return {'text': transcript['text']}

def main():
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level ="info")

if __name__ == "__main__":
#    main()
    pass



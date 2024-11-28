wget https://openaiassets.blob.core.windows.net/\$web/API/docs/audio/alloy.wav
curl http://localhost:8000/v1/audio/transcriptions   -H "Content-Type: multipart/form-data"   -F model="whisper-1"   -F file="@alloy.wav"   -F response_format=text

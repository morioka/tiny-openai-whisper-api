# 環境変数からAPIキーを取得
$apiKey = $env:OPENAI_API_KEY
if (-not $apiKey) {
    Write-Error "環境変数 'OPENAI_API_KEY' が設定されていません。APIキーを設定してください。"
    exit 1
}

# 環境変数からAPI URLを取得
$apiBase = $env:OPENAI_BASE_URL
if (-not $apiBase) {
    $apiBase = $env:OPENAI_API_BASE
    if (-not $apiBase) {
        $apiBase = "https://api.openai.com/v1"
    }
}
$openAiUrl = $apiBase + "/audio/transcriptions"
#$openAiUrl = "http://localhost:8000/v1/audio/transcriptions"

# コマンドライン引数で音声ファイルのパスを取得
if ($args.Count -lt 1) {
    Write-Error "音声ファイルのパスを指定してください。"
    Write-Host "使用例: .\TranscribeAudio.ps1 path\to\audio.mp4"
    exit 1
}
$audioFilePath = $args[0]
#$audioFilePath = "c:\temp\alloy.wav"  # MP4ファイルのパスを指定

# ファイルの存在を確認
if (-Not (Test-Path -Path $audioFilePath)) {
    Write-Error "指定したファイルが見つかりません: $audioFilePath"
    exit 1
}

# ファイルの拡張子を確認し、MIME タイプを設定
$extension = [System.IO.Path]::GetExtension($audioFilePath).ToLower()
switch ($extension) {
    ".mp4" { $mimeType = "audio/mp4" }
    ".wav" { $mimeType = "audio/wav" }
    ".mp3" { $mimeType = "audio/mpeg" }
    default {
        Write-Error "サポートされていないファイル形式: $extension"
        exit 1
    }
}

# マルチパートリクエスト用データを作成
$boundary = [System.Guid]::NewGuid().ToString()
$LF = "`r`n"

$fileBytes = [System.IO.File]::ReadAllBytes($audioFilePath)
$fileContent = [System.Text.Encoding]::GetEncoding("ISO-8859-1").GetString($fileBytes)
$fileName = [System.IO.Path]::GetFileName($audioFilePath)

$body = (
  "--$boundary$LF" + 
  "Content-Disposition: form-data; name=`"file`"; filename=`"$fileName`"$LF" +
  "Content-Type: $mimeType$LF$LF" +
  $fileContent + $LF +
  "--$boundary$LF" +
  "Content-Disposition: form-data; name=`"model`"$LF$LF" +
  "whisper-1$LF" +
  "--$boundary--$LF"
)

$headers = @{
    "Authorization" = "Bearer $apiKey"
    "Content-Type"  = "multipart/form-data; boundary=$boundary"
}

# APIを呼び出す
try {
    $response = Invoke-RestMethod -Uri $openAiUrl -Headers $headers -Method Post -Body $body
    # UTF-8でコンソールに出力する
    #[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    #Write-Output $response.text

    $garbledText = $response.text
    # 文字化けしている文字列をバイト配列に変換
    $byteArray = [System.Text.Encoding]::GetEncoding("ISO-8859-1").GetBytes($garbledText)
    # バイト配列をUTF-8としてデコード
    $decodedText = [System.Text.Encoding]::UTF8.GetString($byteArray)

    Write-Output $decodedText

} catch {
    Write-Error "API呼び出し中にエラーが発生しました: $_"
}

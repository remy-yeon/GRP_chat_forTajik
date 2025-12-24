#!/bin/bash

echo "=============================================="
echo "  Tajikistan RAG Server (WSL)"
echo "=============================================="

cd "$(dirname "$0")"

if [ -d ".venv" ]; then
    echo "[1/3] Activating venv..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "[1/3] Activating venv..."
    source venv/bin/activate
else
    echo "[1/3] Creating venv..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi

echo "[2/3] Ensuring Ollama is running..."
if command -v ollama >/dev/null 2>&1; then
    if ! pgrep -x "ollama" >/dev/null; then
        ollama serve >/dev/null 2>&1 &
        sleep 2
    fi

    if ! ollama list | grep -q "gemma2:2b"; then
        ollama pull gemma2:2b
    fi
else
    echo "  Ollama not found in WSL. Install it or run it on Windows."
fi

echo "[3/3] Starting FastAPI..."
echo "  http://localhost:8000"
echo "  http://localhost:8000/docs"

if command -v cmd.exe >/dev/null 2>&1; then
    ( sleep 3; cmd.exe /c start http://localhost:8000 >/dev/null 2>&1 ) &
fi

LOG_FILE="server.log"
uvicorn llm:app --host 0.0.0.0 --port 8000 2>&1 | tee "$LOG_FILE"

echo ""
echo "서버가 종료되었습니다. 로그: $LOG_FILE"
read -n 1 -s -r -p "아무 키나 누르면 닫힙니다..."

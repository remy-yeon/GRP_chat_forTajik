## Mac
### Ollama + Gemma2:2b 다운로드 명령어


# Homebrew 설치 방식(1,2 중 아무거나 하면됨)
1)
brew install ollama
ollama serve
2)
curl -fsSL https://ollama.com/install.sh | sh

# Gemma2:2b 모델 다운로드 및 실행
ollama run gemma2:2b

# 다운로드만
ollama pull gemma2:2b

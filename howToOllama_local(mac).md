# 🖥️ macOS에서 Ollama + Gemma 2 2B 사용 가이드

> **환경 기준** - OS: macOS (Intel / Apple Silicon M1·M2·M3 모두 가능) -
> LLM: Ollama + `gemma2:2b` - 터미널: macOS 기본 터미널 또는 iTerm2

------------------------------------------------------------------------

# 1. 전체 구성 개요

Mac에서는 Windows처럼 WSL이 필요 없으며, **macOS에 직접 Ollama를
설치**하는 것이 가장 간단하고 안정적이다.

구성 흐름:

1.  macOS에 **Ollama 설치**
2.  Gemma2:2b **모델 다운로드**
3.  `ollama run`으로 채팅 테스트
4.  (선택) Python에서 Ollama API 호출

------------------------------------------------------------------------

# 2. macOS 사전 준비

### macOS 버전 확인

``` bash
sw_vers
```

### Homebrew 설치 여부 확인

``` bash
brew --version
```

없다면 설치:

``` bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

------------------------------------------------------------------------

# 3. macOS에서 Ollama 설치

macOS에서는 두 가지 방법 중 하나만 실행하면 된다.

### 방법 1) Homebrew 설치

``` bash
brew install ollama
```

설치 후 서비스 실행 확인:

``` bash
ollama serve
```

### 방법 2) 공식 스크립트 설치

``` bash
curl -fsSL https://ollama.com/install.sh | sh
```

설치 확인:

``` bash
ollama --version
```

------------------------------------------------------------------------

# 4. Gemma 2 2B 모델 다운로드

### 4.1 Gemma2:2b 모델 다운로드

``` bash
ollama pull gemma2:2b
```

### 4.2 설치된 모델 목록 확인

``` bash
ollama list
```

------------------------------------------------------------------------

# 5. Gemma 2 2B 실행 (채팅)

``` bash
ollama run gemma2:2b
```

예시:

    >>> hello
    Hello! How can I help you today?

종료: `Ctrl + C`

------------------------------------------------------------------------

# 6. Python에서 Ollama API 사용하기 (선택)

### 6.1 pip 패키지 설치

``` bash
pip3 install requests
```

### 6.2 테스트 스크립트 작성

``` bash
nano test_gemma.py
```

내용 입력:

``` python
import requests

url = "http://localhost:11434/api/chat"

payload = {
    "model": "gemma2:2b",
    "messages": [
        {
            "role": "user",
            "content": "한 문장으로 RAG를 쉽게 설명해줘."
        }
    ],
    "stream": False
}

print("🔁 요청 보내는 중...")
resp = requests.post(url, json=payload)
resp.raise_for_status()
data = resp.json()

print("\n🧠 모델 응답:")
print(data["message"]["content"])
```

### 6.3 실행

``` bash
python3 test_gemma.py
```

------------------------------------------------------------------------

# 7. 요약

-   ✔ macOS에서는 WSL 없이 바로 Ollama 설치 가능\

-   ✔ 모델 이름은 `gemma2:2b`\

-   ✔ 실행 명령:

    ``` bash
    ollama run gemma2:2b
    ```

-   ✔ API 엔드포인트:

        http://localhost:11434

-   ✔ Python 예제는 requests로 POST 요청

------------------------------------------------------------------------

필요에 따라 본 문서는 GitHub README 용으로 바로 사용 가능합니다.

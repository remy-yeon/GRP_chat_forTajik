# Windows에서 WSL + Ollama + Gemma 2 2B 사용 가이드

> 환경 기준
>
> * OS: Windows 10/11
> * WSL2 + Ubuntu
> * LLM: Ollama + `gemma2:2b`

---

## 1. 전체 구성 개요

Windows에서는 직접 Ollama를 돌리기보다 **WSL2 안의 Ubuntu에서 Ollama를 실행**하는 방식이 가장 안정적이다.

구성 흐름:

1. Windows에서 **WSL2 + Ubuntu 준비**
2. Ubuntu 안에서 **Ollama 설치**
3. `gemma2:2b` 모델 다운로드
4. `ollama run`으로 대화 테스트
5. (선택) Python으로 로컬 API 호출

---

## 2. WSL2 + Ubuntu 상태 확인 (Windows 기준)

### 2.1 WSL 배포판 목록 확인

Windows에서 **PowerShell / Windows Terminal** 열고:

```powershell
wsl -l -v
```

예시 출력:

```text
  NAME              STATE           VERSION
* Ubuntu            Running         2
  docker-desktop    Running         2
```

* `Ubuntu` 가 존재하고 `VERSION`이 `2`이면 사용 준비 완료.
* `wsl --install -d Ubuntu` 실행 시
  `ERROR_ALREADY_EXISTS` 가 뜨는 것은 **이미 설치되어 있다는 의미**로 정상이다.

### 2.2 Ubuntu 셸 진입

PowerShell / Windows Terminal에서:

```powershell
wsl -d Ubuntu
```

프롬프트 예:

```bash
ubuntu_user@HOSTNAME:~$
```

이제부터는 **Ubuntu 리눅스 환경**에서 명령을 실행한다.

---

## 3. Ollama 설치 (Ubuntu 내부)

Ubuntu 셸에서:

### 3.1 시스템 업데이트 및 필수 패키지 설치

```bash
sudo apt update
sudo apt install -y curl git python3 python3-pip
```

### 3.2 Ollama 설치

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

설치가 끝나면 로그에 다음과 같이 표시된다:

```text
>>> The Ollama API is now available at 127.0.0.1:11434.
>>> Install complete. Run "ollama" from the command line.
```

Ollama 버전 확인:

```bash
ollama --version
```

---

## 4. Gemma 2 2B 모델 다운로드

### 4.1 Ollama 서버 실행

Ubuntu에서 Ollama는 systemd 서비스로 자동 실행되지만, 일반적으로 다음만 확인하면 된다:

```bash
ps aux | grep ollama
```

또는 단순히 `ollama pull`이 동작하면 서버는 정상이라고 보면 된다.

### 4.2 Gemma 2 2B 모델 받기

```bash
ollama pull gemma2:2b
```

정상일 경우:

```text
pulling manifest
pulling 7462734796d6: 100% ...
...
success
```

설치된 모델 목록 확인:

```bash
ollama list
```

예시:

```text
NAME        ID        SIZE     MODIFIED
gemma2:2b   ...       1.6 GB   ...
```

---

## 5. Gemma 2 2B 실행 (채팅 모드)

Ubuntu 셸에서:

```bash
ollama run gemma2:2b
```

프롬프트가 `>>>` 형태로 바뀐다.

예시:

```text
>>> hello
Hello! 👋  How can I help you today? 😄

>>> 너는 할 수 있는 언어가 몇개니?
(모델 응답 출력)
```

원하는 질문을 입력하며 대화하면 되고,
종료는 `Ctrl + C` 로 한다.

---

## 6. Python에서 Ollama API 사용 (선택)

### 6.1 `requests` 설치

Ubuntu 시스템 Python은 PEP 668 때문에 `pip install`이 제한되므로,
가장 간단하게 **APT 패키지**로 설치한다.

```bash
sudo apt install -y python3-requests
```

### 6.2 테스트 스크립트 작성

원하는 작업 디렉터리로 이동 (예: Windows 홈 디렉터리):

```bash
cd /mnt/c/Users/USERNAME
```

파일 생성:

```bash
nano test_gemma.py
```

다음 내용을 그대로 입력/붙여넣기:

```python
import requests

url = "http://localhost:11434/api/chat"

payload = {
    "model": "gemma2:2b",
    "messages": [
        {
            "role": "user",
            "content": "한 문장으로 RAG가 무엇인지 아주 쉽게 설명해줘."
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

저장 후 종료:

* `Ctrl + O` → Enter
* `Ctrl + X`

### 6.3 실행

```bash
python3 test_gemma.py
```

정상 실행 시:

* `🔁 요청 보내는 중...`
* 그 아래에 `🧠 모델 응답:` 과 함께 Gemma 2의 답변이 출력된다.

---

## 7. 정리

* **Windows에서 직접 설치 X → WSL2 + Ubuntu에서 Ollama 설치**

* Gemma 2 2B 모델 이름은 `gemma2:2b`

* 대화용 실행:

  ```bash
  ollama run gemma2:2b
  ```

* 프로그램에서 사용 시:

  * Ollama 서버는 `http://localhost:11434`
  * Python에서는 `requests`로 `/api/chat` 엔드포인트에 POST 요청

이 문서를 `ollama_gemma_windows_guide.md` 같은 이름으로 저장해두면,
나중에 환경 다시 세팅할 때 그대로 재사용할 수 있다.

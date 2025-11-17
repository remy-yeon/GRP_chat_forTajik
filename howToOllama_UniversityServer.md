## 🧱 1️ Ollama 설치 관련 명령어

```bash
# 1. Ollama 설치 시도 (sudo 권한 문제 발생)
curl -fsSL https://ollama.com/install.sh | sh
# 결과: "c22100349 is not in the sudoers file." → 수동 설치로 전환

```

---

## 🗂️ 2️ 수동 설치 과정

```bash
# 2. 설치 디렉토리 생성
mkdir -p ~/ollama/bin
cd ~/ollama/bin

# 3. Ollama 바이너리 다운로드
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama.tgz

# 4. 압축 해제
tar -xvzf ollama.tgz
# bin/ollama, lib/ollama/* 등이 생성됨

```

---

## ⚙️ 3 Ollama 실행 권한 설정

```bash
# Ollama 실행 파일에 권한 추가
chmod +x ~/ollama/bin/bin/ollama

```

---

## 🚀 4 Ollama 서버 실행

```bash
nohup ~/ollama/bin/bin/ollama serve > ~/ollama/ollama.log 2>&1 &
```

실행 후 로그 예시:

```
time=2025-11-12T16:41:21.694+09:00 level=INFO source=routes.go msg="Listening on 127.0.0.1:11434"

```

→ ✅ **서버가 정상적으로 11434 포트에서 실행됨**

---

## 💾 5 Gemma 모델 다운로드

```bash
# Ollama 서버가 실행 중인 상태에서 모델 다운로드
~/ollama/bin/bin/ollama pull gemma:2b

```

> ollama pull gemma:2b 는 Gemma 모델을 로컬에 캐싱합니다.
> 
> 
> (`~/.ollama/models/blobs/` 경로에 저장됨)
> 

---

## 🧠 6 모델 테스트

```bash
# 모델이 잘 설치되었는지 확인
~/ollama/bin/bin/ollama run gemma:2b

```

## 실행 예시
> 대한민국의 수도는?
대한민국의 수도는 서울입니다. 서울은 한국의 수도로, 다른 수도는 부산, 광주, 대구, 울산 등이 있습니
다.[GIN] 2025/11/12 - 17:05:29 | 200 |  856.891026ms |       127.0.0.1 | POST     "/api/chat"
> 

> 러시아의 수도는?
러시아의 수도는 모스크바입니다. 모스크바는 러시아의 수도로, 다른 수도는 제산, 란데르, 볼로그라드 등
이 있습니다.[GIN] 2025/11/12 - 17:05:38 | 200 |  819.686906ms |       127.0.0.1 | POST     "/api/chat"
> 

> 타지키스탄 화폐이름은?
타지키스탄 화폐의 이름은 토그르크이다.[GIN] 2025/11/12 - 17:05:53 | 200 |  511.423461ms |       127.0.0.1 | POST     "/api/chat"
> 

> what is the capital of the america
the capital of the America is Washington, D.C.[GIN] 2025/11/12 - 17:06:23 | 200 |  462.843122ms |       127.0.0.1 | POST     "/api/chat"
>

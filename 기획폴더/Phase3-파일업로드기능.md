# Phase 3: 파일 업로드 기능

## 목표
PDF 및 TXT 파일을 업로드하는 기능을 구현합니다. 사용자가 📎 버튼을 클릭하여 파일을 선택하고 서버에 업로드할 수 있습니다.

---

## 1. HTML 구조

### 1.1 파일 입력 요소 (Phase 2에서 이미 추가됨)
```html
<!-- 숨겨진 파일 입력 -->
<input
    type="file"
    id="fileInput"
    accept=".pdf,.txt"
    hidden
>

<!-- 첨부 버튼 -->
<button class="attach-btn" id="attachBtn" aria-label="파일 첨부">
    📎
</button>
```

### 1.2 업로드 진행 표시 (선택적)
```html
<!-- 입력 영역 상단에 추가 -->
<div class="upload-indicator" id="uploadIndicator" style="display: none;">
    <span class="upload-spinner"></span>
    <span class="upload-text">파일 업로드 중...</span>
</div>
```

---

## 2. CSS 스타일

### 2.1 첨부 버튼 스타일 (Phase 2에서 정의됨)
```css
.attach-btn {
    background: none;
    border: none;
    color: var(--text-secondary);
    font-size: 20px;
    cursor: pointer;
    padding: 8px;
    border-radius: 8px;
    transition: background-color 0.2s, color 0.2s;
}

.attach-btn:hover {
    background-color: rgba(255, 255, 255, 0.1);
    color: var(--text-primary);
}

.attach-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}
```

### 2.2 업로드 진행 표시
```css
.upload-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    background-color: rgba(16, 163, 127, 0.1);
    border-radius: 8px;
    margin-bottom: 8px;
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
}

.upload-spinner {
    width: 16px;
    height: 16px;
    border: 2px solid var(--accent);
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    to {
        transform: rotate(360deg);
    }
}

.upload-text {
    color: var(--accent);
    font-size: 14px;
}
```

### 2.3 업로드 성공/실패 알림
```css
.upload-toast {
    position: fixed;
    bottom: 100px;
    left: 50%;
    transform: translateX(-50%);
    padding: 12px 24px;
    border-radius: 8px;
    color: white;
    font-size: 14px;
    z-index: 1000;
    animation: fadeInOut 3s ease-in-out;
}

.upload-toast.success {
    background-color: var(--accent);
}

.upload-toast.error {
    background-color: #E74C3C;
}

@keyframes fadeInOut {
    0%, 100% { opacity: 0; transform: translateX(-50%) translateY(20px); }
    10%, 90% { opacity: 1; transform: translateX(-50%) translateY(0); }
}
```

---

## 3. JavaScript 기능

### 3.1 파일 선택 처리
```javascript
document.addEventListener('DOMContentLoaded', () => {
    const attachBtn = document.getElementById('attachBtn');
    const fileInput = document.getElementById('fileInput');

    // 📎 버튼 클릭 시 파일 선택 창 열기
    attachBtn.addEventListener('click', () => {
        if (!state.isLoading) {
            fileInput.click();
        }
    });

    // 파일 선택 시 업로드 처리
    fileInput.addEventListener('change', handleFileSelect);
});
```

### 3.2 파일 선택 핸들러
```javascript
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    // 파일 형식 검증
    const validTypes = ['application/pdf', 'text/plain'];
    const validExtensions = ['.pdf', '.txt'];
    const fileExtension = file.name.toLowerCase().slice(file.name.lastIndexOf('.'));

    if (!validTypes.includes(file.type) && !validExtensions.includes(fileExtension)) {
        showToast('PDF 또는 TXT 파일만 업로드할 수 있습니다.', 'error');
        event.target.value = ''; // 입력 초기화
        return;
    }

    // 파일 크기 검증 (예: 10MB 제한)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
        showToast('파일 크기는 10MB 이하여야 합니다.', 'error');
        event.target.value = '';
        return;
    }

    // 업로드 진행
    uploadFile(file);

    // 파일 입력 초기화 (같은 파일 다시 선택 가능하게)
    event.target.value = '';
}
```

### 3.3 파일 업로드 함수
```javascript
async function uploadFile(file) {
    const uploadIndicator = document.getElementById('uploadIndicator');

    try {
        // 업로드 상태 표시
        state.isLoading = true;
        if (uploadIndicator) {
            uploadIndicator.style.display = 'flex';
        }
        disableInputs(true);

        // FormData 생성
        const formData = new FormData();
        formData.append('file', file);

        // API 호출
        const response = await fetch('/upload_file', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`업로드 실패: ${response.status}`);
        }

        const result = await response.json();

        // 성공 처리
        showToast(`${file.name} 업로드 완료!`, 'success');

        // 문서 목록 새로고침
        await loadDocuments();

    } catch (error) {
        console.error('파일 업로드 오류:', error);
        showToast(`업로드 실패: ${error.message}`, 'error');
    } finally {
        // 상태 복원
        state.isLoading = false;
        if (uploadIndicator) {
            uploadIndicator.style.display = 'none';
        }
        disableInputs(false);
    }
}
```

### 3.4 입력 비활성화 함수
```javascript
function disableInputs(disabled) {
    const attachBtn = document.getElementById('attachBtn');
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');

    attachBtn.disabled = disabled;
    messageInput.disabled = disabled;
    sendBtn.disabled = disabled;
}
```

### 3.5 토스트 알림 함수
```javascript
function showToast(message, type = 'success') {
    // 기존 토스트 제거
    const existingToast = document.querySelector('.upload-toast');
    if (existingToast) {
        existingToast.remove();
    }

    // 새 토스트 생성
    const toast = document.createElement('div');
    toast.className = `upload-toast ${type}`;
    toast.textContent = message;

    document.body.appendChild(toast);

    // 3초 후 자동 제거
    setTimeout(() => {
        toast.remove();
    }, 3000);
}
```

---

## 4. API 스펙

### 4.1 파일 업로드 엔드포인트

| 항목 | 값 |
|------|-----|
| 엔드포인트 | `POST /upload_file` |
| Content-Type | `multipart/form-data` |
| 요청 필드 | `file`: 업로드할 파일 |

### 4.2 예상 응답
```json
{
    "success": true,
    "doc_id": "abc123",
    "source": "document.pdf",
    "message": "파일이 성공적으로 업로드되었습니다."
}
```

### 4.3 에러 응답
```json
{
    "success": false,
    "error": "지원하지 않는 파일 형식입니다."
}
```

---

## 5. 지원 파일 형식

| 형식 | MIME 타입 | 확장자 |
|------|-----------|--------|
| PDF | application/pdf | .pdf |
| 텍스트 | text/plain | .txt |

---

## 6. 체크리스트

- [ ] 📎 버튼 클릭 시 파일 선택 창 열기
- [ ] PDF/TXT 파일만 선택 가능하도록 필터링
- [ ] 파일 형식 검증
- [ ] 파일 크기 검증 (10MB 이하)
- [ ] FormData로 파일 전송
- [ ] 업로드 진행 표시
- [ ] 성공/실패 토스트 알림
- [ ] 업로드 중 입력 비활성화
- [ ] 업로드 완료 후 문서 목록 새로고침

---

## 7. 사용자 플로우

```
┌─────────────────────────────────────────────────────────┐
│  1. 📎 버튼 클릭                                        │
│         ↓                                               │
│  2. 파일 선택 창 열림                                   │
│         ↓                                               │
│  3. PDF 또는 TXT 파일 선택                              │
│         ↓                                               │
│  4. 파일 형식/크기 검증                                 │
│         ↓                                               │
│  5. 업로드 진행 표시                                    │
│         ↓                                               │
│  6. 서버로 파일 전송 (POST /upload_file)                │
│         ↓                                               │
│  7. 성공/실패 토스트 표시                               │
│         ↓                                               │
│  8. 사이드바 문서 목록 업데이트                         │
└─────────────────────────────────────────────────────────┘
```

---

## 8. 다음 단계
→ **Phase 4: 문서 관리 사이드바**로 진행

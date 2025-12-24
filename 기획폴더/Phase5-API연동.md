# Phase 5: API 연동

## 목표
백엔드 서버와의 모든 API 통신을 구현합니다. 질문 전송, 파일 업로드, 문서 관리 등 모든 엔드포인트를 연동합니다.

---

## 1. API 엔드포인트 총정리

| 액션 | 메서드 | 엔드포인트 | 설명 |
|------|--------|------------|------|
| 파일 업로드 | POST | `/upload_file` | PDF/TXT 파일 업로드 |
| 질문 전송 | POST | `/ask` | 질문하고 답변 받기 |
| 문서 목록 | GET | `/list_documents` | 업로드된 문서 조회 |
| 문서 삭제 | DELETE | `/delete_document?doc_id=` | 개별 문서 삭제 |
| 전체 삭제 | DELETE | `/clear_all` | 모든 문서 삭제 |

---

## 2. API 설정

### 2.1 기본 설정
```javascript
// API 기본 URL 설정 (환경에 따라 변경)
const API_BASE_URL = ''; // 같은 origin인 경우 빈 문자열

// 또는 별도 서버인 경우
// const API_BASE_URL = 'http://localhost:8000';

// 요청 타임아웃 설정 (밀리초)
const REQUEST_TIMEOUT = 30000;
```

### 2.2 공통 fetch 래퍼
```javascript
async function apiRequest(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;

    // 타임아웃 설정
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal,
            headers: {
                ...options.headers
            }
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `HTTP ${response.status}`);
        }

        return response;

    } catch (error) {
        clearTimeout(timeoutId);

        if (error.name === 'AbortError') {
            throw new Error('요청 시간이 초과되었습니다.');
        }

        throw error;
    }
}
```

---

## 3. 질문 전송 API

### 3.1 요청 형식
```javascript
// POST /ask
// Content-Type: application/json

{
    "question": "사용자의 질문 내용"
}
```

### 3.2 응답 형식
```json
{
    "answer": "AI의 답변 내용...",
    "sources": [
        {
            "source": "document.pdf",
            "page": 3,
            "content": "관련 내용 일부..."
        }
    ]
}
```

### 3.3 구현 코드
```javascript
async function sendToAPI(question) {
    try {
        state.isLoading = true;
        disableInputs(true);

        // 로딩 메시지 표시
        showTypingIndicator();

        const response = await apiRequest('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question })
        });

        const data = await response.json();

        // 로딩 표시 제거
        hideTypingIndicator();

        // AI 답변 추가
        const assistantMessage = {
            role: 'assistant',
            content: data.answer,
            sources: data.sources || []
        };

        state.messages.push(assistantMessage);

        // UI에 메시지 렌더링
        const container = document.getElementById('messagesContainer');
        container.appendChild(renderMessage(assistantMessage));

        // 스크롤 맨 아래로
        container.scrollTop = container.scrollHeight;

    } catch (error) {
        console.error('질문 전송 오류:', error);
        hideTypingIndicator();

        // 에러 메시지 표시
        const errorMessage = {
            role: 'assistant',
            content: `오류가 발생했습니다: ${error.message}`,
            isError: true
        };

        state.messages.push(errorMessage);
        const container = document.getElementById('messagesContainer');
        container.appendChild(renderMessage(errorMessage));

    } finally {
        state.isLoading = false;
        disableInputs(false);
    }
}
```

---

## 4. 파일 업로드 API

### 4.1 요청 형식
```javascript
// POST /upload_file
// Content-Type: multipart/form-data

const formData = new FormData();
formData.append('file', fileObject);
```

### 4.2 응답 형식
```json
{
    "success": true,
    "doc_id": "abc123",
    "source": "document.pdf",
    "message": "업로드 완료"
}
```

### 4.3 구현 코드 (Phase 3에서 정의)
```javascript
async function uploadFile(file) {
    const uploadIndicator = document.getElementById('uploadIndicator');

    try {
        state.isLoading = true;
        if (uploadIndicator) uploadIndicator.style.display = 'flex';
        disableInputs(true);

        const formData = new FormData();
        formData.append('file', file);

        const response = await apiRequest('/upload_file', {
            method: 'POST',
            body: formData
            // Content-Type은 자동 설정됨 (multipart/form-data)
        });

        const result = await response.json();

        showToast(`${file.name} 업로드 완료!`, 'success');
        await loadDocuments();

    } catch (error) {
        console.error('파일 업로드 오류:', error);
        showToast(`업로드 실패: ${error.message}`, 'error');

    } finally {
        state.isLoading = false;
        if (uploadIndicator) uploadIndicator.style.display = 'none';
        disableInputs(false);
    }
}
```

---

## 5. 문서 목록 API

### 5.1 요청/응답
```javascript
// GET /list_documents

// 응답
{
    "documents": [
        { "doc_id": "abc123", "source": "file1.pdf" },
        { "doc_id": "def456", "source": "file2.txt" }
    ]
}
```

### 5.2 구현 코드
```javascript
async function loadDocuments() {
    try {
        const response = await apiRequest('/list_documents', {
            method: 'GET'
        });

        const data = await response.json();
        state.documents = data.documents || [];
        renderDocumentsList();

    } catch (error) {
        console.error('문서 목록 로드 오류:', error);
        // 실패해도 UI는 유지
    }
}
```

---

## 6. 문서 삭제 API

### 6.1 개별 삭제
```javascript
// DELETE /delete_document?doc_id=abc123

async function deleteDocument(docId, fileName) {
    if (!confirm(`"${fileName}"을(를) 삭제하시겠습니까?`)) {
        return;
    }

    try {
        await apiRequest(`/delete_document?doc_id=${encodeURIComponent(docId)}`, {
            method: 'DELETE'
        });

        state.documents = state.documents.filter(doc => doc.doc_id !== docId);
        renderDocumentsList();
        showToast(`${fileName} 삭제됨`, 'success');

    } catch (error) {
        console.error('문서 삭제 오류:', error);
        showToast('삭제 실패', 'error');
    }
}
```

### 6.2 전체 삭제
```javascript
// DELETE /clear_all

async function clearAllDocuments() {
    if (state.documents.length === 0) return;

    if (!confirm('모든 문서를 삭제하시겠습니까?')) {
        return;
    }

    try {
        await apiRequest('/clear_all', {
            method: 'DELETE'
        });

        state.documents = [];
        renderDocumentsList();
        showToast('모든 문서가 삭제되었습니다', 'success');

    } catch (error) {
        console.error('전체 삭제 오류:', error);
        showToast('전체 삭제 실패', 'error');
    }
}
```

---

## 7. 에러 처리

### 7.1 에러 유형별 처리
```javascript
function handleApiError(error, context) {
    let message = '알 수 없는 오류가 발생했습니다.';

    if (error.message.includes('Failed to fetch')) {
        message = '서버에 연결할 수 없습니다. 네트워크를 확인해주세요.';
    } else if (error.message.includes('시간이 초과')) {
        message = '요청 시간이 초과되었습니다. 다시 시도해주세요.';
    } else if (error.message.includes('HTTP 400')) {
        message = '잘못된 요청입니다.';
    } else if (error.message.includes('HTTP 404')) {
        message = '요청한 리소스를 찾을 수 없습니다.';
    } else if (error.message.includes('HTTP 500')) {
        message = '서버 오류가 발생했습니다.';
    } else if (error.message) {
        message = error.message;
    }

    console.error(`[${context}] API 오류:`, error);
    return message;
}
```

### 7.2 재시도 로직 (선택적)
```javascript
async function apiRequestWithRetry(endpoint, options = {}, maxRetries = 3) {
    let lastError;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            return await apiRequest(endpoint, options);
        } catch (error) {
            lastError = error;

            // 클라이언트 에러는 재시도하지 않음
            if (error.message.includes('HTTP 4')) {
                throw error;
            }

            if (attempt < maxRetries) {
                // 지수 백오프
                await new Promise(resolve =>
                    setTimeout(resolve, Math.pow(2, attempt) * 1000)
                );
            }
        }
    }

    throw lastError;
}
```

---

## 8. 타이핑 인디케이터

### 8.1 HTML
```html
<div class="typing-indicator" id="typingIndicator" style="display: none;">
    <div class="message assistant">
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    </div>
</div>
```

### 8.2 CSS
```css
.typing-dots {
    display: flex;
    gap: 4px;
    padding: 8px 0;
}

.typing-dots span {
    width: 8px;
    height: 8px;
    background-color: var(--text-secondary);
    border-radius: 50%;
    animation: typing 1.4s infinite ease-in-out;
}

.typing-dots span:nth-child(1) { animation-delay: 0s; }
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }

@keyframes typing {
    0%, 60%, 100% {
        transform: translateY(0);
        opacity: 0.4;
    }
    30% {
        transform: translateY(-4px);
        opacity: 1;
    }
}
```

### 8.3 JavaScript
```javascript
function showTypingIndicator() {
    const container = document.getElementById('messagesContainer');
    const indicator = document.getElementById('typingIndicator');

    if (indicator) {
        indicator.style.display = 'block';
        container.appendChild(indicator);
        container.scrollTop = container.scrollHeight;
    }
}

function hideTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) {
        indicator.style.display = 'none';
    }
}
```

---

## 9. 체크리스트

- [ ] API 기본 URL 설정
- [ ] 공통 fetch 래퍼 함수 구현
- [ ] 요청 타임아웃 처리
- [ ] 질문 전송 API 연동 (POST /ask)
- [ ] 파일 업로드 API 연동 (POST /upload_file)
- [ ] 문서 목록 API 연동 (GET /list_documents)
- [ ] 개별 문서 삭제 API 연동 (DELETE /delete_document)
- [ ] 전체 삭제 API 연동 (DELETE /clear_all)
- [ ] 에러 처리 및 사용자 피드백
- [ ] 타이핑 인디케이터 구현
- [ ] 네트워크 오류 처리

---

## 10. 테스트 시나리오

1. **정상 플로우**
   - 파일 업로드 → 문서 목록에 표시
   - 질문 전송 → 답변 수신 및 표시
   - 문서 삭제 → 목록에서 제거

2. **에러 케이스**
   - 서버 미연결 상태에서 요청
   - 대용량 파일 업로드 시도
   - 타임아웃 발생

3. **동시성**
   - 업로드 중 질문 전송 시도
   - 빠른 연속 요청 처리

---

## 11. 다음 단계
→ **Phase 6: 반응형 디자인**으로 진행

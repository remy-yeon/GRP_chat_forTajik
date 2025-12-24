# Phase 7: 로딩 상태 및 출처 표시

## 목표
AI 답변 생성 중 타이핑 애니메이션을 표시하고, 답변 하단에 참조한 문서/페이지 출처를 표시합니다.

---

## 1. 로딩 상태 (타이핑 인디케이터)

### 1.1 HTML 구조
```html
<!-- 타이핑 인디케이터 (messagesContainer 내부에 동적 추가) -->
<div class="message assistant typing-message" id="typingIndicator">
    <div class="message-avatar">🤖</div>
    <div class="message-content">
        <div class="typing-dots">
            <span></span>
            <span></span>
            <span></span>
        </div>
    </div>
</div>
```

### 1.2 CSS 스타일
```css
/* 타이핑 인디케이터 */
.typing-message {
    opacity: 0.7;
}

.typing-dots {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 0;
}

.typing-dots span {
    width: 8px;
    height: 8px;
    background-color: var(--text-secondary);
    border-radius: 50%;
    animation: typingBounce 1.4s infinite ease-in-out;
}

.typing-dots span:nth-child(1) {
    animation-delay: 0s;
}

.typing-dots span:nth-child(2) {
    animation-delay: 0.2s;
}

.typing-dots span:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes typingBounce {
    0%, 60%, 100% {
        transform: translateY(0);
        opacity: 0.4;
    }
    30% {
        transform: translateY(-6px);
        opacity: 1;
    }
}
```

### 1.3 JavaScript 함수
```javascript
// 타이핑 인디케이터 표시
function showTypingIndicator() {
    const container = document.getElementById('messagesContainer');

    // 이미 존재하면 제거 후 다시 추가
    hideTypingIndicator();

    const indicator = document.createElement('div');
    indicator.className = 'message assistant typing-message';
    indicator.id = 'typingIndicator';

    indicator.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;

    container.appendChild(indicator);
    container.scrollTop = container.scrollHeight;
}

// 타이핑 인디케이터 제거
function hideTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) {
        indicator.remove();
    }
}
```

---

## 2. 출처 표시 (Sources)

### 2.1 데이터 구조
```javascript
// API 응답 예시
{
    "answer": "AI의 답변 내용...",
    "sources": [
        {
            "source": "document.pdf",
            "page": 3,
            "content": "관련 내용 미리보기..."
        },
        {
            "source": "notes.txt",
            "page": null,
            "content": "텍스트 파일의 관련 내용..."
        }
    ]
}
```

### 2.2 HTML 구조
```html
<!-- 메시지 내 출처 표시 -->
<div class="message assistant">
    <div class="message-avatar">🤖</div>
    <div class="message-content">
        <p>AI 답변 내용...</p>

        <!-- 출처 섹션 -->
        <div class="sources-section">
            <div class="sources-header">
                <span class="sources-icon">📚</span>
                <span class="sources-title">참조 문서</span>
            </div>
            <div class="sources-list">
                <div class="source-item">
                    <span class="source-file">📕 document.pdf</span>
                    <span class="source-page">p.3</span>
                </div>
                <div class="source-item">
                    <span class="source-file">📄 notes.txt</span>
                </div>
            </div>
        </div>
    </div>
</div>
```

### 2.3 CSS 스타일
```css
/* 출처 섹션 */
.sources-section {
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px solid var(--border-color);
}

.sources-header {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 8px;
    color: var(--text-secondary);
    font-size: 13px;
}

.sources-icon {
    font-size: 14px;
}

.sources-title {
    font-weight: 500;
}

.sources-list {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.source-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    font-size: 13px;
    color: var(--text-primary);
    transition: background-color 0.2s;
}

.source-item:hover {
    background-color: rgba(255, 255, 255, 0.1);
}

.source-file {
    display: flex;
    align-items: center;
    gap: 4px;
}

.source-page {
    color: var(--text-secondary);
    font-size: 12px;
}
```

### 2.4 확장형 출처 (미리보기 포함)
```css
/* 클릭하여 확장 가능한 출처 */
.source-item.expandable {
    cursor: pointer;
    flex-direction: column;
    align-items: flex-start;
}

.source-preview {
    display: none;
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px dashed var(--border-color);
    font-size: 12px;
    color: var(--text-secondary);
    line-height: 1.5;
    max-height: 80px;
    overflow: hidden;
}

.source-item.expanded .source-preview {
    display: block;
}

.source-toggle {
    font-size: 10px;
    color: var(--text-secondary);
    margin-left: auto;
}
```

---

## 3. JavaScript 구현

### 3.1 메시지 렌더링 (출처 포함)
```javascript
function renderMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${message.role}`;

    const avatar = message.role === 'user' ? '👤' : '🤖';

    // 에러 메시지 스타일
    const contentClass = message.isError ? 'message-content error' : 'message-content';

    let html = `
        <div class="message-avatar">${avatar}</div>
        <div class="${contentClass}">
            <p>${formatMessageContent(message.content)}</p>
    `;

    // 출처가 있는 경우 추가
    if (message.sources && message.sources.length > 0) {
        html += renderSources(message.sources);
    }

    html += `</div>`;
    messageDiv.innerHTML = html;

    // 출처 토글 이벤트 바인딩
    if (message.sources && message.sources.length > 0) {
        bindSourceToggleEvents(messageDiv);
    }

    return messageDiv;
}
```

### 3.2 출처 렌더링 함수
```javascript
function renderSources(sources) {
    if (!sources || sources.length === 0) return '';

    const sourceItems = sources.map(source => {
        const icon = source.source.endsWith('.pdf') ? '📕' : '📄';
        const pageInfo = source.page ? `<span class="source-page">p.${source.page}</span>` : '';
        const hasPreview = source.content && source.content.trim();

        if (hasPreview) {
            return `
                <div class="source-item expandable" data-preview="${escapeHtml(source.content)}">
                    <div class="source-header">
                        <span class="source-file">${icon} ${escapeHtml(source.source)}</span>
                        ${pageInfo}
                        <span class="source-toggle">▼</span>
                    </div>
                    <div class="source-preview">${escapeHtml(source.content)}</div>
                </div>
            `;
        } else {
            return `
                <div class="source-item">
                    <span class="source-file">${icon} ${escapeHtml(source.source)}</span>
                    ${pageInfo}
                </div>
            `;
        }
    }).join('');

    return `
        <div class="sources-section">
            <div class="sources-header">
                <span class="sources-icon">📚</span>
                <span class="sources-title">참조 문서 (${sources.length})</span>
            </div>
            <div class="sources-list">
                ${sourceItems}
            </div>
        </div>
    `;
}
```

### 3.3 출처 토글 이벤트
```javascript
function bindSourceToggleEvents(messageDiv) {
    const expandableItems = messageDiv.querySelectorAll('.source-item.expandable');

    expandableItems.forEach(item => {
        item.addEventListener('click', () => {
            item.classList.toggle('expanded');

            const toggle = item.querySelector('.source-toggle');
            if (toggle) {
                toggle.textContent = item.classList.contains('expanded') ? '▲' : '▼';
            }
        });
    });
}
```

### 3.4 메시지 내용 포맷팅
```javascript
function formatMessageContent(content) {
    // HTML 이스케이프
    let formatted = escapeHtml(content);

    // 줄바꿈 처리
    formatted = formatted.replace(/\n/g, '<br>');

    // 코드 블록 처리 (간단한 버전)
    formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');

    return formatted;
}
```

---

## 4. 추가 로딩 상태

### 4.1 버튼 로딩 상태
```css
/* 전송 버튼 로딩 상태 */
.send-btn.loading {
    pointer-events: none;
}

.send-btn.loading::after {
    content: '';
    position: absolute;
    width: 16px;
    height: 16px;
    border: 2px solid white;
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}

.send-btn.loading span {
    visibility: hidden;
}

@keyframes spin {
    to {
        transform: rotate(360deg);
    }
}
```

### 4.2 스켈레톤 로딩 (선택적)
```css
/* 문서 목록 스켈레톤 */
.document-skeleton {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
}

.skeleton-icon {
    width: 20px;
    height: 20px;
    background: linear-gradient(90deg, var(--border-color) 25%, #3a3a3a 50%, var(--border-color) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 4px;
}

.skeleton-text {
    flex: 1;
    height: 14px;
    background: linear-gradient(90deg, var(--border-color) 25%, #3a3a3a 50%, var(--border-color) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 4px;
}

@keyframes shimmer {
    0% {
        background-position: 200% 0;
    }
    100% {
        background-position: -200% 0;
    }
}
```

---

## 5. 에러 상태 표시

### 5.1 에러 메시지 스타일
```css
.message-content.error {
    background-color: rgba(231, 76, 60, 0.1);
    border: 1px solid rgba(231, 76, 60, 0.3);
    color: #E74C3C;
}

.message-content.error p {
    display: flex;
    align-items: flex-start;
    gap: 8px;
}

.message-content.error::before {
    content: '⚠️';
}
```

### 5.2 재시도 버튼
```html
<div class="message-content error">
    <p>오류가 발생했습니다. 다시 시도해 주세요.</p>
    <button class="retry-btn" onclick="retryLastMessage()">
        🔄 다시 시도
    </button>
</div>
```

```css
.retry-btn {
    margin-top: 8px;
    padding: 6px 12px;
    background-color: transparent;
    border: 1px solid #E74C3C;
    border-radius: 6px;
    color: #E74C3C;
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s;
}

.retry-btn:hover {
    background-color: rgba(231, 76, 60, 0.1);
}
```

---

## 6. 전체 상태 관리 통합

```javascript
// 상태 객체 업데이트
const state = {
    messages: [],
    documents: [],
    isLoading: false,
    sidebarOpen: true,
    lastQuestion: null  // 재시도를 위한 마지막 질문 저장
};

// 질문 전송 시 저장
function handleSendMessage() {
    const input = document.getElementById('messageInput');
    const content = input.value.trim();

    if (!content || state.isLoading) return;

    state.lastQuestion = content;  // 마지막 질문 저장

    // ... 나머지 로직
}

// 재시도 함수
function retryLastMessage() {
    if (state.lastQuestion && !state.isLoading) {
        // 마지막 에러 메시지 제거
        const messages = document.querySelectorAll('.message.assistant');
        const lastMessage = messages[messages.length - 1];
        if (lastMessage && lastMessage.querySelector('.error')) {
            lastMessage.remove();
            state.messages.pop();
        }

        // 다시 시도
        sendToAPI(state.lastQuestion);
    }
}
```

---

## 7. 체크리스트

- [ ] 타이핑 인디케이터 HTML 구조
- [ ] 타이핑 애니메이션 CSS
- [ ] showTypingIndicator() 함수
- [ ] hideTypingIndicator() 함수
- [ ] 출처 섹션 HTML 구조
- [ ] 출처 아이템 스타일링
- [ ] renderSources() 함수
- [ ] 출처 토글 (확장/축소) 기능
- [ ] 에러 메시지 스타일
- [ ] 재시도 버튼 구현
- [ ] 마지막 질문 저장 및 재시도 로직

---

## 8. 최종 통합 테스트

### 8.1 테스트 시나리오
1. 질문 전송 → 타이핑 인디케이터 표시
2. 답변 수신 → 인디케이터 제거, 답변 + 출처 표시
3. 출처 클릭 → 미리보기 확장/축소
4. 에러 발생 → 에러 메시지 + 재시도 버튼
5. 재시도 클릭 → 다시 질문 전송

### 8.2 확인 사항
- 타이핑 애니메이션이 부드럽게 동작하는가?
- 출처가 없는 답변도 정상 표시되는가?
- 긴 출처 미리보기가 잘 잘리는가?
- 에러 시 사용자에게 명확한 피드백이 제공되는가?

---

## 9. 완료

이로써 모든 Phase가 완료되었습니다!

### 구현 완료 항목
- ✅ Phase 1: 기본 구조 및 레이아웃
- ✅ Phase 2: 채팅 UI 구현
- ✅ Phase 3: 파일 업로드 기능
- ✅ Phase 4: 문서 관리 사이드바
- ✅ Phase 5: API 연동
- ✅ Phase 6: 반응형 디자인
- ✅ Phase 7: 로딩 상태 및 출처 표시

### 최종 결과물
단일 HTML 파일로 구성된 Ollama RAG Chat 프론트엔드
- Vanilla JS + CSS (외부 라이브러리 없음)
- 다크 테마 (ChatGPT 스타일)
- 반응형 (데스크톱 + 모바일)
- 브라우저에서 바로 실행 가능

# Phase 2: 채팅 UI 구현

## 목표
사용자와 AI 간의 대화를 표시하는 채팅 인터페이스와 메시지 입력 영역을 구현합니다.

---

## 1. HTML 구조

### 1.1 채팅 컨테이너 구조
```html
<main class="chat-container">
    <!-- 헤더 (모바일용 햄버거 메뉴 포함) -->
    <header class="chat-header">
        <button class="menu-toggle" aria-label="메뉴 열기">
            <span class="hamburger-icon">☰</span>
        </button>
        <h1 class="chat-title">Ollama RAG Chat</h1>
    </header>

    <!-- 메시지 표시 영역 -->
    <div class="messages-container" id="messagesContainer">
        <!-- 환영 메시지 (대화 시작 전) -->
        <div class="welcome-message" id="welcomeMessage">
            <h2>무엇을 도와드릴까요?</h2>
            <p>PDF 또는 TXT 파일을 업로드하고 질문해 보세요.</p>
        </div>

        <!-- 메시지들이 여기에 동적으로 추가됨 -->
    </div>

    <!-- 입력 영역 -->
    <div class="input-area">
        <div class="input-wrapper">
            <!-- 파일 첨부 버튼 -->
            <button class="attach-btn" id="attachBtn" aria-label="파일 첨부">
                📎
            </button>
            <input type="file" id="fileInput" accept=".pdf,.txt" hidden>

            <!-- 메시지 입력창 -->
            <textarea
                class="message-input"
                id="messageInput"
                placeholder="메시지를 입력하세요..."
                rows="1"
            ></textarea>

            <!-- 전송 버튼 -->
            <button class="send-btn" id="sendBtn" aria-label="전송">
                ➤
            </button>
        </div>
    </div>
</main>
```

---

## 2. CSS 스타일

### 2.1 채팅 헤더
```css
.chat-header {
    display: flex;
    align-items: center;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color);
    background-color: var(--bg-main);
}

.menu-toggle {
    display: none; /* 데스크톱에서는 숨김 */
    background: none;
    border: none;
    color: var(--text-primary);
    font-size: 24px;
    cursor: pointer;
    padding: 8px;
    margin-right: 12px;
}

.chat-title {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
}
```

### 2.2 메시지 컨테이너
```css
.messages-container {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
}

/* 스크롤바 스타일링 */
.messages-container::-webkit-scrollbar {
    width: 8px;
}

.messages-container::-webkit-scrollbar-track {
    background: transparent;
}

.messages-container::-webkit-scrollbar-thumb {
    background-color: var(--border-color);
    border-radius: 4px;
}
```

### 2.3 환영 메시지
```css
.welcome-message {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    text-align: center;
    color: var(--text-secondary);
}

.welcome-message h2 {
    font-size: 24px;
    color: var(--text-primary);
    margin-bottom: 8px;
}

.welcome-message p {
    font-size: 15px;
}
```

### 2.4 메시지 버블
```css
.message {
    display: flex;
    gap: 12px;
    max-width: 800px;
    margin: 0 auto;
    width: 100%;
}

.message-avatar {
    width: 36px;
    height: 36px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}

.message.user .message-avatar {
    background-color: var(--accent);
}

.message.assistant .message-avatar {
    background-color: #6B5B95;
}

.message-content {
    flex: 1;
    padding: 12px 16px;
    border-radius: 12px;
    line-height: 1.6;
}

.message.user .message-content {
    background-color: var(--user-bubble);
}

.message.assistant .message-content {
    background-color: var(--assistant-bubble);
}
```

### 2.5 입력 영역
```css
.input-area {
    padding: 16px 24px 24px;
    background-color: var(--bg-main);
}

.input-wrapper {
    display: flex;
    align-items: flex-end;
    gap: 8px;
    max-width: 800px;
    margin: 0 auto;
    background-color: var(--bg-input);
    border-radius: 24px;
    padding: 8px 16px;
    border: 1px solid var(--border-color);
}

.input-wrapper:focus-within {
    border-color: var(--accent);
}

.attach-btn {
    background: none;
    border: none;
    color: var(--text-secondary);
    font-size: 20px;
    cursor: pointer;
    padding: 8px;
    border-radius: 8px;
    transition: background-color 0.2s;
}

.attach-btn:hover {
    background-color: rgba(255, 255, 255, 0.1);
    color: var(--text-primary);
}

.message-input {
    flex: 1;
    background: none;
    border: none;
    color: var(--text-primary);
    font-size: 15px;
    resize: none;
    outline: none;
    min-height: 24px;
    max-height: 200px;
    font-family: inherit;
    line-height: 1.5;
}

.message-input::placeholder {
    color: var(--text-secondary);
}

.send-btn {
    background-color: var(--accent);
    border: none;
    color: white;
    width: 36px;
    height: 36px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background-color 0.2s;
}

.send-btn:hover {
    background-color: var(--accent-hover);
}

.send-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}
```

---

## 3. JavaScript 기능

### 3.1 상태 관리
```javascript
const state = {
    messages: [],
    documents: [],
    isLoading: false,
    sidebarOpen: true
};
```

### 3.2 메시지 렌더링 함수
```javascript
function renderMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${message.role}`;

    const avatar = message.role === 'user' ? '👤' : '🤖';

    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">${escapeHtml(message.content)}</div>
    `;

    return messageDiv;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
```

### 3.3 메시지 전송 처리
```javascript
function handleSendMessage() {
    const input = document.getElementById('messageInput');
    const content = input.value.trim();

    if (!content || state.isLoading) return;

    // 사용자 메시지 추가
    const userMessage = { role: 'user', content };
    state.messages.push(userMessage);

    // UI 업데이트
    const container = document.getElementById('messagesContainer');
    const welcomeMsg = document.getElementById('welcomeMessage');
    if (welcomeMsg) welcomeMsg.style.display = 'none';

    container.appendChild(renderMessage(userMessage));

    // 입력창 초기화
    input.value = '';
    autoResizeTextarea(input);

    // 스크롤 맨 아래로
    container.scrollTop = container.scrollHeight;

    // API 호출 (Phase 5에서 구현)
    sendToAPI(content);
}
```

### 3.4 텍스트 영역 자동 크기 조절
```javascript
function autoResizeTextarea(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
}
```

### 3.5 이벤트 리스너 설정
```javascript
document.addEventListener('DOMContentLoaded', () => {
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');

    // 전송 버튼 클릭
    sendBtn.addEventListener('click', handleSendMessage);

    // Enter 키로 전송 (Shift+Enter는 줄바꿈)
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });

    // 텍스트 영역 자동 크기 조절
    messageInput.addEventListener('input', () => {
        autoResizeTextarea(messageInput);
    });
});
```

---

## 4. 체크리스트

- [ ] 채팅 헤더 구현
- [ ] 메시지 표시 영역 구현
- [ ] 환영 메시지 표시
- [ ] 사용자 메시지 버블 스타일
- [ ] AI 답변 버블 스타일
- [ ] 입력창 구현 (textarea)
- [ ] 파일 첨부 버튼 (📎) 배치
- [ ] 전송 버튼 구현
- [ ] Enter 키 전송 기능
- [ ] 텍스트 영역 자동 크기 조절
- [ ] 스크롤 자동 이동

---

## 5. UI 미리보기

```
┌────────────────────────────────────┐
│         Ollama RAG Chat            │
├────────────────────────────────────┤
│                                    │
│  👤 ┌─────────────────────────┐   │
│     │ 사용자 질문 메시지      │   │
│     └─────────────────────────┘   │
│                                    │
│  🤖 ┌─────────────────────────┐   │
│     │ AI 답변 메시지          │   │
│     └─────────────────────────┘   │
│                                    │
├────────────────────────────────────┤
│ ┌──────────────────────────────┐  │
│ │ 📎 │ 메시지를 입력하세요... │➤│  │
│ └──────────────────────────────┘  │
└────────────────────────────────────┘
```

---

## 6. 다음 단계
→ **Phase 3: 파일 업로드 기능**으로 진행

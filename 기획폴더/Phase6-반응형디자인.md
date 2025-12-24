# Phase 6: 반응형 디자인

## 목표
데스크톱과 모바일 환경 모두에서 최적화된 사용자 경험을 제공하는 반응형 레이아웃을 구현합니다.

---

## 1. 브레이크포인트 정의

```css
/* 브레이크포인트 */
/* 모바일: < 768px */
/* 데스크톱: ≥ 768px */

:root {
    --breakpoint-mobile: 768px;
}
```

---

## 2. 레이아웃 비교

### 데스크톱 (≥768px)
```
┌──────────┬────────────────────┐
│ 사이드바  │                    │
│          │                    │
│ 📁 문서   │    채팅 영역        │
│ ────────│                    │
│ doc1.pdf│  👤 질문 메시지      │
│ doc2.txt│  🤖 답변 메시지      │
│          │                    │
│ [전체삭제]│                    │
├──────────┼────────────────────┤
│          │ 📎 │ 메시지 입력.. │➤│
└──────────┴────────────────────┘
```

### 모바일 (<768px)
```
┌────────────────────┐
│ ☰  Ollama RAG Chat │
├────────────────────┤
│                    │
│    채팅 영역        │
│                    │
│  👤 질문           │
│  🤖 답변           │
│                    │
├────────────────────┤
│ 📎 │ 메시지 입력.. │➤│
└────────────────────┘

햄버거 메뉴 클릭 시:
┌──────────────┬─────┐
│ 📁 문서     ✕│     │
│              │     │
│ doc1.pdf   ✕ │ 오버│
│ doc2.txt   ✕ │ 레이│
│              │     │
│ [전체삭제]   │     │
└──────────────┴─────┘
```

---

## 3. CSS 미디어 쿼리

### 3.1 모바일 기본 스타일
```css
/* 모바일 우선 접근법 (Mobile First) */

/* 사이드바 - 모바일에서 숨김 처리 */
.sidebar {
    position: fixed;
    top: 0;
    left: -100%;
    width: 80%;
    max-width: 300px;
    height: 100vh;
    z-index: 100;
    transition: left 0.3s ease;
}

.sidebar.open {
    left: 0;
}

/* 사이드바 닫기 버튼 표시 */
.sidebar-close {
    display: block;
}

/* 햄버거 메뉴 표시 */
.menu-toggle {
    display: flex;
}

/* 오버레이 */
.sidebar-overlay {
    display: none;
}

.sidebar-overlay.active {
    display: block;
}

/* 채팅 컨테이너 전체 너비 */
.chat-container {
    width: 100%;
}

/* 메시지 패딩 조정 */
.messages-container {
    padding: 16px;
}

/* 입력 영역 패딩 조정 */
.input-area {
    padding: 12px 16px 16px;
}
```

### 3.2 데스크톱 스타일
```css
@media (min-width: 768px) {
    /* 사이드바 - 항상 표시 */
    .sidebar {
        position: relative;
        left: 0;
        width: 260px;
    }

    /* 사이드바 닫기 버튼 숨김 */
    .sidebar-close {
        display: none;
    }

    /* 햄버거 메뉴 숨김 */
    .menu-toggle {
        display: none;
    }

    /* 오버레이 비활성화 */
    .sidebar-overlay {
        display: none !important;
    }

    /* 메시지 패딩 복원 */
    .messages-container {
        padding: 24px;
    }

    /* 입력 영역 패딩 복원 */
    .input-area {
        padding: 16px 24px 24px;
    }
}
```

---

## 4. 모바일 사이드바 제어

### 4.1 JavaScript 함수
```javascript
// 사이드바 열기
function openSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebarOverlay');

    sidebar.classList.add('open');
    overlay.classList.add('active');
    state.sidebarOpen = true;

    // 스크롤 방지
    document.body.style.overflow = 'hidden';
}

// 사이드바 닫기
function closeSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebarOverlay');

    sidebar.classList.remove('open');
    overlay.classList.remove('active');
    state.sidebarOpen = false;

    // 스크롤 복원
    document.body.style.overflow = '';
}

// 사이드바 토글
function toggleSidebar() {
    if (state.sidebarOpen) {
        closeSidebar();
    } else {
        openSidebar();
    }
}
```

### 4.2 이벤트 리스너
```javascript
document.addEventListener('DOMContentLoaded', () => {
    const menuToggle = document.getElementById('menuToggle');
    const sidebarClose = document.getElementById('sidebarClose');
    const sidebarOverlay = document.getElementById('sidebarOverlay');

    // 햄버거 메뉴 클릭
    if (menuToggle) {
        menuToggle.addEventListener('click', toggleSidebar);
    }

    // 사이드바 닫기 버튼
    if (sidebarClose) {
        sidebarClose.addEventListener('click', closeSidebar);
    }

    // 오버레이 클릭 시 닫기
    if (sidebarOverlay) {
        sidebarOverlay.addEventListener('click', closeSidebar);
    }

    // ESC 키로 닫기
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && state.sidebarOpen) {
            closeSidebar();
        }
    });
});
```

---

## 5. 화면 크기 변경 처리

### 5.1 리사이즈 이벤트
```javascript
// 화면 크기 변경 감지
function handleResize() {
    const isDesktop = window.innerWidth >= 768;

    if (isDesktop) {
        // 데스크톱으로 전환 시 사이드바 상태 초기화
        const sidebar = document.getElementById('sidebar');
        const overlay = document.getElementById('sidebarOverlay');

        sidebar.classList.remove('open');
        overlay.classList.remove('active');
        document.body.style.overflow = '';
        state.sidebarOpen = true; // 데스크톱에서는 항상 열림
    } else {
        // 모바일로 전환 시 사이드바 닫기
        state.sidebarOpen = false;
    }
}

// 디바운스 처리
let resizeTimeout;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(handleResize, 150);
});

// 초기 상태 설정
handleResize();
```

---

## 6. 터치 제스처 지원 (선택적)

### 6.1 스와이프로 사이드바 열기
```javascript
let touchStartX = 0;
let touchEndX = 0;

document.addEventListener('touchstart', (e) => {
    touchStartX = e.changedTouches[0].screenX;
}, { passive: true });

document.addEventListener('touchend', (e) => {
    touchEndX = e.changedTouches[0].screenX;
    handleSwipe();
}, { passive: true });

function handleSwipe() {
    const swipeDistance = touchEndX - touchStartX;
    const minSwipeDistance = 50;

    // 오른쪽 스와이프로 사이드바 열기 (왼쪽 가장자리에서)
    if (swipeDistance > minSwipeDistance && touchStartX < 30) {
        if (window.innerWidth < 768) {
            openSidebar();
        }
    }

    // 왼쪽 스와이프로 사이드바 닫기
    if (swipeDistance < -minSwipeDistance && state.sidebarOpen) {
        closeSidebar();
    }
}
```

---

## 7. 반응형 타이포그래피

```css
/* 기본 (모바일) */
html {
    font-size: 14px;
}

.chat-title {
    font-size: 16px;
}

.welcome-message h2 {
    font-size: 20px;
}

/* 데스크톱 */
@media (min-width: 768px) {
    html {
        font-size: 15px;
    }

    .chat-title {
        font-size: 18px;
    }

    .welcome-message h2 {
        font-size: 24px;
    }
}
```

---

## 8. 반응형 메시지 레이아웃

```css
/* 메시지 최대 너비 조정 */
.message {
    max-width: 100%;
}

@media (min-width: 768px) {
    .message {
        max-width: 800px;
        margin: 0 auto;
    }
}

/* 입력 영역 최대 너비 */
.input-wrapper {
    max-width: 100%;
}

@media (min-width: 768px) {
    .input-wrapper {
        max-width: 800px;
        margin: 0 auto;
    }
}
```

---

## 9. 키보드 대응 (모바일)

### 9.1 가상 키보드 표시 시 레이아웃 조정
```javascript
// iOS Safari 대응
if (/iPhone|iPad|iPod/.test(navigator.userAgent)) {
    const messageInput = document.getElementById('messageInput');

    messageInput.addEventListener('focus', () => {
        // 키보드 표시 시 스크롤 조정
        setTimeout(() => {
            messageInput.scrollIntoView({ behavior: 'smooth', block: 'end' });
        }, 300);
    });
}
```

### 9.2 CSS 조정
```css
/* 모바일에서 입력창 포커스 시 */
@media (max-width: 767px) {
    .input-area {
        position: sticky;
        bottom: 0;
        background-color: var(--bg-main);
    }

    /* iOS 안전 영역 대응 */
    @supports (padding-bottom: env(safe-area-inset-bottom)) {
        .input-area {
            padding-bottom: calc(16px + env(safe-area-inset-bottom));
        }
    }
}
```

---

## 10. 체크리스트

- [ ] 768px 브레이크포인트 설정
- [ ] 모바일: 사이드바 숨김 처리
- [ ] 모바일: 햄버거 메뉴(☰) 표시
- [ ] 모바일: 사이드바 슬라이드 애니메이션
- [ ] 모바일: 오버레이 처리
- [ ] 모바일: 사이드바 닫기 버튼(✕)
- [ ] 데스크톱: 사이드바 항상 표시
- [ ] 데스크톱: 햄버거 메뉴 숨김
- [ ] 화면 크기 변경 시 레이아웃 전환
- [ ] 터치 스와이프 제스처 (선택)
- [ ] iOS 안전 영역 대응
- [ ] 모바일 키보드 대응

---

## 11. 테스트 체크포인트

| 항목 | 모바일 | 데스크톱 |
|------|--------|----------|
| 사이드바 | 햄버거 메뉴로 열림 | 항상 표시 |
| 오버레이 | 사이드바 열릴 때 표시 | 없음 |
| 입력창 | 전체 너비 | 최대 800px |
| 메시지 | 전체 너비 | 최대 800px, 중앙 정렬 |
| 폰트 크기 | 14px | 15px |

---

## 12. 다음 단계
→ **Phase 7: 로딩 상태 및 출처 표시**로 진행

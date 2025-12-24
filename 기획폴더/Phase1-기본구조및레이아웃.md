# Phase 1: 기본 구조 및 레이아웃

## 목표
HTML 기본 골격과 CSS 레이아웃 시스템을 구축하여 전체 애플리케이션의 뼈대를 완성합니다.

---

## 1. HTML 기본 구조

### 1.1 문서 설정
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ollama RAG Chat</title>
    <style>
        /* CSS가 여기에 위치 */
    </style>
</head>
<body>
    <div class="app-container">
        <!-- 사이드바 -->
        <aside class="sidebar">
            <!-- Phase 4에서 구현 -->
        </aside>

        <!-- 메인 채팅 영역 -->
        <main class="chat-container">
            <!-- Phase 2에서 구현 -->
        </main>
    </div>

    <script>
        /* JavaScript가 여기에 위치 */
    </script>
</body>
</html>
```

---

## 2. CSS 기본 스타일

### 2.1 CSS 리셋 및 기본 설정
```css
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body {
    height: 100%;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 15px;
    line-height: 1.5;
}
```

### 2.2 다크 테마 색상 변수
```css
:root {
    /* 배경 색상 */
    --bg-main: #212121;
    --bg-sidebar: #171717;
    --bg-input: #2F2F2F;

    /* 텍스트 색상 */
    --text-primary: #ECECEC;
    --text-secondary: #8E8E8E;

    /* 액센트 색상 */
    --accent: #10A37F;
    --accent-hover: #0D8A6A;

    /* 테두리 및 구분선 */
    --border-color: #3E3E3E;

    /* 기타 */
    --user-bubble: #2F2F2F;
    --assistant-bubble: transparent;
}
```

### 2.3 레이아웃 구조
```css
body {
    background-color: var(--bg-main);
    color: var(--text-primary);
}

.app-container {
    display: flex;
    height: 100vh;
    overflow: hidden;
}

/* 사이드바 기본 스타일 */
.sidebar {
    width: 260px;
    background-color: var(--bg-sidebar);
    border-right: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
}

/* 메인 채팅 영역 */
.chat-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0; /* flexbox 오버플로우 방지 */
}
```

---

## 3. 디자인 스펙 요약

| 요소 | 값 |
|------|-----|
| 테마 | 다크 모드 (ChatGPT 스타일) |
| 메인 배경 | #212121 |
| 사이드바 배경 | #171717 |
| 기본 텍스트 | #ECECEC |
| 보조 텍스트 | #8E8E8E |
| 액센트 색상 | #10A37F |
| 폰트 | system-ui |
| 본문 크기 | 15px |
| 메시지 버블 모서리 | 12px |
| 입력창 모서리 | 24px |

---

## 4. 체크리스트

- [ ] HTML 문서 기본 구조 생성
- [ ] meta 태그 설정 (viewport, charset)
- [ ] CSS 리셋 적용
- [ ] CSS 변수로 색상 팔레트 정의
- [ ] Flexbox 레이아웃 구성
- [ ] 사이드바 기본 영역 설정
- [ ] 메인 채팅 영역 기본 설정

---

## 5. 예상 결과물

이 Phase를 완료하면 다음과 같은 화면이 나타납니다:
- 왼쪽에 어두운 사이드바 영역 (260px 너비)
- 오른쪽에 메인 채팅 영역
- 다크 테마 적용
- 전체 화면 높이 사용 (100vh)

---

## 6. 다음 단계
→ **Phase 2: 채팅 UI 구현**으로 진행

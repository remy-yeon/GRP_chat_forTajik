# Phase 4: 문서 관리 사이드바

## 목표
업로드된 문서 목록을 사이드바에 표시하고, 개별 문서 삭제 및 전체 삭제 기능을 구현합니다.

---

## 1. HTML 구조

### 1.1 사이드바 전체 구조
```html
<aside class="sidebar" id="sidebar">
    <!-- 사이드바 헤더 -->
    <div class="sidebar-header">
        <h2 class="sidebar-title">📁 문서</h2>
        <button class="sidebar-close" id="sidebarClose" aria-label="사이드바 닫기">
            ✕
        </button>
    </div>

    <!-- 문서 목록 -->
    <div class="documents-list" id="documentsList">
        <!-- 문서가 없을 때 -->
        <div class="empty-state" id="emptyState">
            <p>업로드된 문서가 없습니다.</p>
            <p class="empty-hint">📎 버튼으로 PDF/TXT 파일을 추가하세요.</p>
        </div>

        <!-- 문서 항목들이 여기에 동적으로 추가됨 -->
    </div>

    <!-- 사이드바 푸터 -->
    <div class="sidebar-footer">
        <button class="clear-all-btn" id="clearAllBtn">
            🗑️ 전체 삭제
        </button>
    </div>
</aside>

<!-- 모바일용 오버레이 -->
<div class="sidebar-overlay" id="sidebarOverlay"></div>
```

### 1.2 문서 항목 템플릿
```html
<!-- 각 문서 항목 구조 -->
<div class="document-item" data-doc-id="abc123">
    <span class="document-icon">📄</span>
    <span class="document-name" title="document.pdf">document.pdf</span>
    <button class="document-delete" aria-label="삭제">✕</button>
</div>
```

---

## 2. CSS 스타일

### 2.1 사이드바 기본 스타일
```css
.sidebar {
    width: 260px;
    background-color: var(--bg-sidebar);
    border-right: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    height: 100vh;
}

.sidebar-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px;
    border-bottom: 1px solid var(--border-color);
}

.sidebar-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
}

.sidebar-close {
    display: none; /* 데스크톱에서는 숨김 */
    background: none;
    border: none;
    color: var(--text-secondary);
    font-size: 18px;
    cursor: pointer;
    padding: 4px 8px;
    border-radius: 4px;
}

.sidebar-close:hover {
    background-color: rgba(255, 255, 255, 0.1);
    color: var(--text-primary);
}
```

### 2.2 문서 목록 스타일
```css
.documents-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
}

.documents-list::-webkit-scrollbar {
    width: 6px;
}

.documents-list::-webkit-scrollbar-thumb {
    background-color: var(--border-color);
    border-radius: 3px;
}

.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    text-align: center;
    color: var(--text-secondary);
    padding: 24px;
}

.empty-state p {
    margin-bottom: 8px;
}

.empty-hint {
    font-size: 13px;
    opacity: 0.7;
}
```

### 2.3 문서 항목 스타일
```css
.document-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    border-radius: 8px;
    cursor: default;
    transition: background-color 0.2s;
}

.document-item:hover {
    background-color: rgba(255, 255, 255, 0.05);
}

.document-icon {
    font-size: 16px;
    flex-shrink: 0;
}

.document-name {
    flex: 1;
    font-size: 14px;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.document-delete {
    opacity: 0;
    background: none;
    border: none;
    color: var(--text-secondary);
    font-size: 14px;
    cursor: pointer;
    padding: 4px 8px;
    border-radius: 4px;
    transition: opacity 0.2s, background-color 0.2s;
}

.document-item:hover .document-delete {
    opacity: 1;
}

.document-delete:hover {
    background-color: rgba(231, 76, 60, 0.2);
    color: #E74C3C;
}
```

### 2.4 사이드바 푸터
```css
.sidebar-footer {
    padding: 12px;
    border-top: 1px solid var(--border-color);
}

.clear-all-btn {
    width: 100%;
    padding: 10px;
    background-color: transparent;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    color: var(--text-secondary);
    font-size: 14px;
    cursor: pointer;
    transition: all 0.2s;
}

.clear-all-btn:hover {
    background-color: rgba(231, 76, 60, 0.1);
    border-color: #E74C3C;
    color: #E74C3C;
}

.clear-all-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}
```

### 2.5 모바일 오버레이
```css
.sidebar-overlay {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.5);
    z-index: 99;
}

.sidebar-overlay.active {
    display: block;
}
```

---

## 3. JavaScript 기능

### 3.1 문서 목록 로드
```javascript
async function loadDocuments() {
    try {
        const response = await fetch('/list_documents');

        if (!response.ok) {
            throw new Error('문서 목록을 불러올 수 없습니다.');
        }

        const data = await response.json();
        state.documents = data.documents || [];

        renderDocumentsList();

    } catch (error) {
        console.error('문서 목록 로드 오류:', error);
    }
}
```

### 3.2 문서 목록 렌더링
```javascript
function renderDocumentsList() {
    const container = document.getElementById('documentsList');
    const emptyState = document.getElementById('emptyState');
    const clearAllBtn = document.getElementById('clearAllBtn');

    // 기존 문서 항목 제거 (emptyState 제외)
    const existingItems = container.querySelectorAll('.document-item');
    existingItems.forEach(item => item.remove());

    if (state.documents.length === 0) {
        // 문서가 없을 때
        if (emptyState) emptyState.style.display = 'flex';
        if (clearAllBtn) clearAllBtn.disabled = true;
    } else {
        // 문서가 있을 때
        if (emptyState) emptyState.style.display = 'none';
        if (clearAllBtn) clearAllBtn.disabled = false;

        state.documents.forEach(doc => {
            const item = createDocumentItem(doc);
            container.appendChild(item);
        });
    }
}
```

### 3.3 문서 항목 생성
```javascript
function createDocumentItem(doc) {
    const item = document.createElement('div');
    item.className = 'document-item';
    item.dataset.docId = doc.doc_id;

    // 파일 확장자에 따른 아이콘
    const icon = doc.source.endsWith('.pdf') ? '📕' : '📄';

    item.innerHTML = `
        <span class="document-icon">${icon}</span>
        <span class="document-name" title="${escapeHtml(doc.source)}">${escapeHtml(doc.source)}</span>
        <button class="document-delete" aria-label="${doc.source} 삭제">✕</button>
    `;

    // 삭제 버튼 이벤트
    const deleteBtn = item.querySelector('.document-delete');
    deleteBtn.addEventListener('click', () => deleteDocument(doc.doc_id, doc.source));

    return item;
}
```

### 3.4 개별 문서 삭제
```javascript
async function deleteDocument(docId, fileName) {
    if (!confirm(`"${fileName}"을(를) 삭제하시겠습니까?`)) {
        return;
    }

    try {
        const response = await fetch(`/delete_document?doc_id=${encodeURIComponent(docId)}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            throw new Error('문서 삭제에 실패했습니다.');
        }

        // 목록에서 제거
        state.documents = state.documents.filter(doc => doc.doc_id !== docId);
        renderDocumentsList();

        showToast(`${fileName} 삭제됨`, 'success');

    } catch (error) {
        console.error('문서 삭제 오류:', error);
        showToast('삭제 실패', 'error');
    }
}
```

### 3.5 전체 문서 삭제
```javascript
async function clearAllDocuments() {
    if (state.documents.length === 0) return;

    if (!confirm('모든 문서를 삭제하시겠습니까?\n이 작업은 되돌릴 수 없습니다.')) {
        return;
    }

    try {
        const response = await fetch('/clear_all', {
            method: 'DELETE'
        });

        if (!response.ok) {
            throw new Error('전체 삭제에 실패했습니다.');
        }

        // 상태 초기화
        state.documents = [];
        renderDocumentsList();

        showToast('모든 문서가 삭제되었습니다', 'success');

    } catch (error) {
        console.error('전체 삭제 오류:', error);
        showToast('전체 삭제 실패', 'error');
    }
}
```

### 3.6 이벤트 리스너 설정
```javascript
document.addEventListener('DOMContentLoaded', () => {
    const clearAllBtn = document.getElementById('clearAllBtn');

    // 전체 삭제 버튼
    if (clearAllBtn) {
        clearAllBtn.addEventListener('click', clearAllDocuments);
    }

    // 페이지 로드 시 문서 목록 불러오기
    loadDocuments();
});
```

---

## 4. API 스펙

### 4.1 문서 목록 조회

| 항목 | 값 |
|------|-----|
| 엔드포인트 | `GET /list_documents` |
| 트리거 | 페이지 로드, 업로드 완료 시 |

**응답 예시:**
```json
{
    "documents": [
        { "doc_id": "abc123", "source": "report.pdf" },
        { "doc_id": "def456", "source": "notes.txt" }
    ]
}
```

### 4.2 개별 문서 삭제

| 항목 | 값 |
|------|-----|
| 엔드포인트 | `DELETE /delete_document?doc_id={id}` |
| 트리거 | 삭제 버튼 클릭 |

### 4.3 전체 문서 삭제

| 항목 | 값 |
|------|-----|
| 엔드포인트 | `DELETE /clear_all` |
| 트리거 | 전체 삭제 버튼 클릭 |

---

## 5. 체크리스트

- [ ] 사이드바 레이아웃 구현
- [ ] 문서 목록 영역 구현
- [ ] 빈 상태 메시지 표시
- [ ] 문서 항목 렌더링
- [ ] 파일 형식별 아이콘 표시 (PDF/TXT)
- [ ] 긴 파일명 말줄임(...) 처리
- [ ] 호버 시 삭제 버튼 표시
- [ ] 개별 문서 삭제 기능
- [ ] 전체 삭제 버튼 구현
- [ ] 삭제 확인 다이얼로그
- [ ] 문서 없을 때 전체 삭제 버튼 비활성화
- [ ] 페이지 로드 시 문서 목록 자동 로드

---

## 6. UI 미리보기

```
┌─────────────────────┐
│ 📁 문서          ✕  │
├─────────────────────┤
│                     │
│ 📕 report.pdf    ✕  │
│ 📄 notes.txt     ✕  │
│ 📕 manual.pdf    ✕  │
│                     │
│                     │
│                     │
├─────────────────────┤
│  🗑️ 전체 삭제       │
└─────────────────────┘
```

---

## 7. 다음 단계
→ **Phase 5: API 연동**으로 진행

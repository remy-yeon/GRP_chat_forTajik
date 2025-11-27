document.getElementById('file').addEventListener('change', function() {
    const fileInput = this;
    const fileNameDisplay = document.getElementById('file-name');

    // Check if a file is selected
    if (fileInput.files.length > 0) {
        // 파일 이름을 업데이트
        fileNameDisplay.textContent = fileInput.files[0].name;
    } else {
        fileNameDisplay.textContent = '선택된 파일 없음';
        }
    });

document.getElementById('qaForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const formData = new FormData(this);
    const fileInput = document.getElementById('file');
    const fileNameDisplay = document.getElementById('file-name');

    const questionInput = document.getElementById('question');
    const spinner = document.getElementById('spinner'); // 스피너 요소 가져오기
    
    // 파일과 질문 값 확인 - 디버깅용 
    console.log('Selected file:', fileInput.files[0]);
    console.log('Question:', questionInput.value);

    formData.append('file', fileInput.files[0]);
    formData.append('question', questionInput.value);

    // FormData 확인 - 디버깅용 
    console.log('FormData file:', formData.get('file'));
    console.log('FormData question:', formData.get('question'));


    spinner.style.display = 'block';
    
    console.log('FormData:', formData);

    // Check if a file is selected
    if (fileInput.files.length === 0) {
        console.log('파일이 선택되지 않았습니다.');
        fileNameDisplay.textContent = '선택된 파일 없음';
    } else {
        console.log('선택된 파일:', fileInput.files[0].name);
        fileNameDisplay.textContent = fileInput.files[0].name; // 파일 이름 표시
    }
    // fetch("http://127.0.0.1:8080/upload", { // URL 수정: '/upload'로 변경
    fetch("/upload", {        
        method: 'POST',
        body: formData,
        headers: {
            'Accept': 'application/json' // 응답 형식 설정
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })    
    .then(data => {
        // Hide the spinner after receiving response
        spinner.style.display = 'none';
        // Show loading complete message
        
        // Display the answer
        document.getElementById("answer").innerText = "답변: " + data.answer;

        // Optionally display the time taken
        document.getElementById("timeTaken").innerText = "답변 생성 시간: " + data.time_taken.toFixed(2) + "초";

    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('answer').textContent = '오류가 발생했습니다.';
        
        // Hide the spinner in case of error
        spinner.style.display = 'none';
    });
});
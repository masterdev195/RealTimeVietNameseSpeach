const fileInput = document.getElementById('audioFile');
const uploadBtn = document.getElementById('uploadBtn');
const statusBox = document.getElementById('status');
const resultList = document.getElementById('resultList');

function setStatus(message, isError = false) {
    statusBox.textContent = message;
    statusBox.className = isError
        ? 'mt-4 text-center text-sm font-semibold text-rose-500'
        : 'mt-4 text-center text-sm font-semibold text-[#2f9a5a]';
}

function renderSegments(segments) {
    resultList.innerHTML = '';

    if (!segments || segments.length === 0) {
        resultList.innerHTML = '<li class="ghibli-card p-5 rounded-2xl text-slate-500 italic text-center">Không tìm thấy lời thì thầm nào.</li>';
        return;
    }

    segments.forEach((item) => {
        const row = document.createElement('li');
        // Tạo style giống bong bóng hội thoại mềm mại
        row.className = 'ghibli-card p-6 rounded-[1.5rem] border-l-8 border-[#ffb9da] transform hover:scale-[1.01] transition-transform';

        const lang = (item.language || 'VN').toUpperCase();

        row.innerHTML = `
            <div class="flex items-center gap-2 mb-2">
                <span class="bg-[#ffb9da] text-[#c83b7c] text-[10px] px-2 py-0.5 rounded-full font-bold">
                    ${lang}
                </span>
                <span class="text-xs text-slate-400 font-medium">✨ Đã ghi nhận</span>
            </div>
            <p class="text-[#2d3748] text-lg leading-relaxed font-medium">${item.text}</p>
        `;

        resultList.appendChild(row);
    });
}

// Logic Fetch API giữ nguyên như bản gốc của bạn[cite: 2]
uploadBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) {
        setStatus('Ôi, bạn chưa chọn file âm thanh kìa! ✨', true);
        return;
    }

    uploadBtn.disabled = true;
    uploadBtn.classList.add('opacity-50');
    setStatus('Gió đang mang âm thanh đi xử lý... vui lòng đợi nhé! 🍃');

    try {
        const formData = new FormData();
        formData.append('audio_file', file);

        const response = await fetch('http://localhost:8000/transcribe-file', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Lỗi xử lý');

        renderSegments(data.segments);
        setStatus(`Xong rồi! Khu vườn đã nghe được ${data.total_segments || 0} câu thoại. ✨`);
    } catch (error) {
        setStatus(`Có chút trục trặc: ${error.message}`, true);
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.classList.remove('opacity-50');
    }
});
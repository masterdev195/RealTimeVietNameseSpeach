const fileInput = document.getElementById('audioFile');
const uploadBtn = document.getElementById('uploadBtn');
const statusBox = document.getElementById('status');
const resultList = document.getElementById('resultList');

function formatTime(seconds) {
    const sec = Math.max(0, Math.floor(seconds));
    const mm = String(Math.floor(sec / 60)).padStart(2, '0');
    const ss = String(sec % 60).padStart(2, '0');
    return `${mm}:${ss}`;
}

function setStatus(message, isError = false) {
    statusBox.textContent = message;
    statusBox.className = isError
        ? 'mt-3 text-sm text-red-300'
        : 'mt-3 text-sm text-emerald-300';
}

function renderSegments(segments) {
    resultList.innerHTML = '';

    if (!segments || segments.length === 0) {
        resultList.innerHTML = '<li class="p-4 rounded-lg bg-zinc-900 border border-zinc-800 text-zinc-400">Không có nội dung nhận diện.</li>';
        return;
    }

    segments.forEach((item) => {
        const row = document.createElement('li');
        row.className = 'p-4 rounded-xl bg-zinc-900 border border-zinc-800';

        const langLabel = item.language && item.language !== 'unknown'
            ? item.language.toUpperCase()
            : (item.detected_language || 'AUTO').toUpperCase();

        row.innerHTML = `
            <div class="flex flex-wrap gap-3 items-center mb-2">
                <span class="px-2.5 py-1 rounded-full bg-emerald-600/20 text-emerald-200 border border-emerald-500/40 text-xs font-semibold">${langLabel}</span>
                <span class="text-xs text-zinc-400">${formatTime(item.start)} - ${formatTime(item.end)}</span>
            </div>
            <p class="text-zinc-100 leading-relaxed">${item.text}</p>
        `;

        resultList.appendChild(row);
    });
}

uploadBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) {
        setStatus('Vui lòng chọn file âm thanh trước.', true);
        return;
    }

    uploadBtn.disabled = true;
    uploadBtn.classList.add('opacity-60', 'cursor-not-allowed');
    setStatus('Đang tải file và xử lý nhận diện...');

    try {
        const formData = new FormData();
        formData.append('audio_file', file);

        const response = await fetch('http://localhost:8000/transcribe-file', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || 'Không thể xử lý file.');
        }

        const segments = data.segments || [];
        renderSegments(segments);

        setStatus(`Hoàn tất: ${data.total_segments || 0} đoạn thoại.`);
    } catch (error) {
        setStatus(`Lỗi: ${error.message}`, true);
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.classList.remove('opacity-60', 'cursor-not-allowed');
    }
});
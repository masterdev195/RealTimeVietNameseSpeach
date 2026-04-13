let socket;
let audioContext;
let recorder;

// Xử lý tải video YouTube từ link
document.getElementById('loadVideo').onclick = () => {
    const url = document.getElementById('videoUrl').value;
    const videoId = url.split('v=')[1]?.split('&')[0];
    if (videoId) {
        document.getElementById('player').innerHTML = `
            <iframe width="100%" height="100%" 
                src="https://www.youtube.com/embed/${videoId}?autoplay=1&mute=0" 
                frameborder="0" allow="autoplay; encrypted-media" allowfullscreen>
            </iframe>`;
    }
};

startBtn.onclick = async () => {
    // 1. Kết nối WebSocket
    socket = new WebSocket('ws://localhost:8000/ws');

    // 2. Lấy âm thanh từ Hệ thống (Tab Audio)
    // Trình duyệt sẽ hiện bảng chọn: Hãy chọn "Tab này" hoặc "Tab YouTube" và tích vào "Share Audio"
    const stream = await navigator.mediaDevices.getDisplayMedia({
        video: true, // Phải để true để hiện bảng chọn, nhưng ta chỉ lấy audio
        audio: {
            echoCancellation: true,
            noiseSuppression: true
        }
    });

    // Chỉ lấy track Audio, dừng track Video để tiết kiệm tài nguyên
    const audioTrack = stream.getAudioTracks()[0];
    stream.getVideoTracks()[0].stop(); 

    audioContext = new AudioContext({ sampleRate: 16000 });
    const source = audioContext.createMediaStreamSource(new MediaStream([audioTrack]));
    
    recorder = audioContext.createScriptProcessor(4096, 1, 1);
    recorder.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        const int16Data = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
            int16Data[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7FFF;
        }
        if (socket.readyState === WebSocket.OPEN) {
            socket.send(int16Data.buffer);
        }
    };

    source.connect(recorder);
    recorder.connect(audioContext.destination);

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        data.forEach(item => {
            const display = document.getElementById('sub-display');
            display.innerText = item.text;
            display.style.opacity = 1;
            
            // Tự ẩn sau khi nói xong
            setTimeout(() => {
                if (display.innerText === item.text) display.style.opacity = 0;
            }, 4000);
        });
    };
};
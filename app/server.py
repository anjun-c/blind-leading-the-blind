from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import time
import shutil
import cv2
import numpy as np
import io
import av  # pip install av
import asyncio
from src.stt.testbyttoaudio import transcribe_audio
from src.cnn.model_run import model_build, live_face_prediction

app = FastAPI()

# Load the CNN model
try:
    model = model_build()
    model.load_weights('src/cnn/model_weights_fer2013.weights.h5')
except Exception as e:
    print("Error: Could not load model", e)

# Allow cross-origin requests if needed
origins = ["http://localhost:8000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files on /static
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Serve index.html at the root
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

# Ensure upload directories exist
if not os.path.exists('uploads/voice'):
    os.makedirs('uploads/voice')
if not os.path.exists('uploads/video'):
    os.makedirs('uploads/video')

# Existing file upload endpoints (unchanged)
@app.post("/api/record_voice")
async def record_voice(file: UploadFile = File(...)):
    if file.filename == "":
        raise HTTPException(status_code=400, detail="No selected file")
    filename = f"voice_{int(time.time())}.webm"
    filepath = os.path.join('uploads', 'voice', filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "Voice recording saved successfully", "filename": filename}

@app.post("/api/record_video")
async def record_video(file: UploadFile = File(...)):
    if file.filename == "":
        raise HTTPException(status_code=400, detail="No selected file")
    filename = f"video_{int(time.time())}.webm"
    filepath = os.path.join('uploads', 'video', filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "Video recording saved successfully", "filename": filename}

# WebSocket endpoint for audio streaming using PyAV for conversion
@app.websocket("/ws/stream_audio")
async def stream_audio(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = bytearray()
    transcription_interval = 3  # seconds
    last_transcription_time = time.time()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)
            
            current_time = time.time()
            if current_time - last_transcription_time >= transcription_interval:
                try:
                    # Wrap the accumulated WebM data in a BytesIO object
                    input_buffer = io.BytesIO(audio_buffer)
                    # Open the container using PyAV
                    input_container = av.open(input_buffer, format='webm')
                    
                    # Prepare an output buffer for WAV data
                    output_buffer = io.BytesIO()
                    output_container = av.open(output_buffer, mode='w', format='wav')
                    
                    # Locate the first audio stream
                    audio_stream = next((s for s in input_container.streams if s.type == 'audio'), None)
                    if audio_stream is None:
                        raise Exception("No audio stream found in input.")
                    
                    # Add an output audio stream using PCM S16LE codec
                    output_audio_stream = output_container.add_stream('pcm_s16le', rate=audio_stream.rate)
                    
                    for frame in input_container.decode(audio=0):
                        frame.pts = None  # reset timestamps
                        packet = output_audio_stream.encode(frame)
                        if packet is not None:
                            output_container.mux(packet)
                    
                    # Flush remaining packets
                    packet = output_audio_stream.encode(None)
                    if packet is not None:
                        output_container.mux(packet)
                    
                    output_container.close()
                    
                    # Get the WAV data from the output buffer
                    wav_data = output_buffer.getvalue()
                    
                    # Write WAV data to temporary file for transcription
                    temp_wav_filepath = os.path.join('uploads', 'voice', "temp_audio.wav")
                    with open(temp_wav_filepath, "wb") as wav_file:
                        wav_file.write(wav_data)
                except Exception as e:
                    await websocket.send_text("Error converting audio with PyAV: " + str(e))
                    print("Error converting audio with PyAV:", e)
                    audio_buffer = bytearray()
                    last_transcription_time = current_time
                    continue

                try:
                    transcription = transcribe_audio(temp_wav_filepath)
                    print("Transcription:", transcription)
                except Exception as e:
                    transcription = "Error transcribing audio."
                    print("Transcription error:", e)
                
                await websocket.send_text(transcription)
                last_transcription_time = current_time
                audio_buffer = bytearray()
    except WebSocketDisconnect:
        print("Audio stream disconnected")

# Helper function to process a single video frame (unchanged)
def process_frame(frame, model):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    label_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_roi = gray[y:y+h, x:x+w]
        try:
            face_resized = cv2.resize(face_roi, (48, 48))
        except Exception as e:
            continue
        face_normalized = face_resized.astype("float32") / 255.0
        input_img = np.expand_dims(face_normalized, axis=0)
        input_img = np.expand_dims(input_img, axis=-1)
        prediction = model.predict(input_img)
        pred_class = np.argmax(prediction, axis=1)[0]
        pred_label = label_map.get(pred_class, "Unknown")
        cv2.putText(frame, f"Prediction: {pred_label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        print(f"Prediction: {pred_label}")
    return frame

@app.websocket("/ws/stream_video")
async def stream_video(websocket: WebSocket):
    await websocket.accept()
    filename = f"stream_video_{int(time.time())}.webm"
    filepath = os.path.join('uploads', 'video', filename)
    
    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            processed_frame = process_frame(frame, model)
            ret, encoded_img = cv2.imencode('.jpg', processed_frame)
            if ret:
                await websocket.send_bytes(encoded_img.tobytes())
    except WebSocketDisconnect:
        print("Video stream disconnected")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

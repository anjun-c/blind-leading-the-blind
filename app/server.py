from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import time
import shutil

app = FastAPI()

# Allow cross-origin requests if needed
origins = [
    "http://localhost:8000",  # Adjust if your frontend runs on a different port or domain
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # You can set to ["*"] for development only
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

# Existing file upload endpoints for audio and video recordings
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

# websocket
@app.websocket("/ws/stream_audio")
async def stream_audio(websocket: WebSocket):
    await websocket.accept()
    filename = f"stream_audio_{int(time.time())}.webm"
    filepath = os.path.join('uploads', 'voice', filename)
    with open(filepath, "wb") as audio_file:
        try:
            while True:
                # Expecting binary data from the client
                data = await websocket.receive_bytes()
                audio_file.write(data)
        except WebSocketDisconnect:
            print("Audio stream disconnected")

@app.websocket("/ws/stream_video")
async def stream_video(websocket: WebSocket):
    await websocket.accept()
    filename = f"stream_video_{int(time.time())}.webm"
    filepath = os.path.join('uploads', 'video', filename)
    with open(filepath, "wb") as video_file:
        try:
            while True:
                # Expecting binary data from the client
                data = await websocket.receive_bytes()
                video_file.write(data)
        except WebSocketDisconnect:
            print("Video stream disconnected")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

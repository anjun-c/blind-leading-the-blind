
# from fastapi import FastAPI, File, UploadFile
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse
# import os
# import shutil

# app = FastAPI()

# # Directory to store uploaded audio files
# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # File to save streamed audio
# AUDIO_FILE = os.path.join(UPLOAD_DIR, "recorded_audio.wav")

# # Mount templates directory for frontend files
# app.mount("/templates", StaticFiles(directory="templates"), name="templates")

# ### 1️⃣ HOMEPAGE ENDPOINT ###
# @app.get("/")
# def homepage():
#     return {"message": "Welcome to the FastAPI audio streaming service!"}

# ### 2️⃣ STREAM AUDIO FROM CLIENT ###
# @app.post("/stream/audio")
# async def stream_audio(audio_chunk: UploadFile = File(...)):
#     with open(AUDIO_FILE, "ab") as f:  # Append mode for streaming chunks
#         shutil.copyfileobj(audio_chunk.file, f)
#     return {"message": "Audio chunk received"}

# ### 3️⃣ DOWNLOAD RECORDED AUDIO ###
# @app.get("/download/audio")
# async def download_audio():
#     if os.path.exists(AUDIO_FILE):
#         return FileResponse(AUDIO_FILE, media_type="audio/wav", filename="recorded_audio.wav")
#     return {"error": "No recorded audio found"}

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
import shutil
import wave

app = FastAPI()

# Create the uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Define the path to the recorded audio file
AUDIO_FILE = os.path.join(UPLOAD_DIR, "recorded_audio.wav")

### 1️⃣ HOMEPAGE ENDPOINT ###
@app.get("/")
def homepage():
    return {"message": "Welcome to the FastAPI audio recording service!"}

### 2️⃣ STREAM AUDIO FROM CLIENT ###
@app.post("/stream/audio")
async def stream_audio(audio_chunk: UploadFile = File(...)):
    """Receives audio chunks and appends them to a WAV file."""
    print(f"Receiving chunk: {audio_chunk.filename}")  # Debugging

    with open(AUDIO_FILE, "ab") as f:  # Append mode
        shutil.copyfileobj(audio_chunk.file, f)

    return {"message": "Audio chunk received"}

### 3️⃣ DOWNLOAD RECORDED AUDIO ###
@app.get("/download/audio")
async def download_audio():
    """Allows the client to download the final recorded audio file."""
    if os.path.exists(AUDIO_FILE):
        return FileResponse(AUDIO_FILE, media_type="audio/wav", filename="recorded_audio.wav")
    return {"error": "No recorded audio found"}

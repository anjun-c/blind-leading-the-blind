from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import time
import shutil

app = FastAPI()

# Optional: if you plan to allow cross-origin requests (e.g., during development or if hosting separately)
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

# Optionally, you can provide a separate root route for a welcome message or redirection
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

# Ensure upload directories exist
if not os.path.exists('uploads/voice'):
    os.makedirs('uploads/voice')
if not os.path.exists('uploads/video'):
    os.makedirs('uploads/video')

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


import subprocess
from fastapi.responses import JSONResponse

@app.post("/api/send_beeps")
async def send_beeps():
    try:
        # Run the beep script located at src\stt\test2.py
        subprocess.run(["python", "src/stt/test2.py"], check=True)
        
        return JSONResponse(content={"message": "Beeps sent successfully"}, status_code=200)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error executing beep script: {str(e)}")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

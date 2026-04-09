import os
import shutil
import uuid
import edge_tts
import google.generativeai as genai
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from typing import Annotated, Optional
from app.services.wav2lip_service import generate_lip_sync

router = APIRouter()

def remove_file(path: str):
    """Helper function to remove a file from the disk securely."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

@router.post("/generate-script")
async def generate_script_ai(
    topic: Annotated[str, Form(description="Topic to generate a script for")],
):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set on the server.")
        
    if not topic.strip():
        raise HTTPException(status_code=400, detail="Please provide a topic.")
        
    try:
        genai.configure(api_key=api_key)
        # Using Gemini 2.5 Flash for ultra-fast text generation
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"Write a short, engaging, and natural-sounding script for an AI avatar video about the following topic. Do not include stage directions, sound effects, or speaker names, just the raw spoken text:\n\n{topic}"
        
        response = model.generate_content(prompt)
        return {"script": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-audio")
async def generate_audio_preview(
    script_text: Annotated[str, Form(description="Text script to generate speech")],
    voice_gender: Annotated[Optional[str], Form(description="Voice gender for TTS")] = "female",
):
    if not script_text.strip():
        raise HTTPException(status_code=400, detail="Please provide a text script.")
        
    os.makedirs("data/temp", exist_ok=True)
    audio_path = f"data/temp/{uuid.uuid4().hex}.wav"
    
    try:
        voice = "en-US-GuyNeural" if voice_gender.lower() == "male" else "en-US-JennyNeural"
        communicate = edge_tts.Communicate(script_text, voice)
        await communicate.save(audio_path)
        
        return FileResponse(
            audio_path, 
            media_type="audio/wav", 
            filename="preview.wav",
            background=BackgroundTask(remove_file, audio_path)
        )
    except Exception as e:
        remove_file(audio_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate")
async def generate_video(
    request: Request,
    face_file: Annotated[UploadFile, File(description="Image or Video file of the face")],
    audio_file: Annotated[Optional[UploadFile], File(description="Audio file containing speech")] = None,
    script_text: Annotated[Optional[str], Form(description="Text script to generate speech")] = None,
    voice_gender: Annotated[Optional[str], Form(description="Voice gender for TTS")] = "female",
):
    model = request.app.state.wav2lip_model
    
    if not audio_file and not script_text:
        raise HTTPException(status_code=400, detail="Please provide either an audio file or a text script.")
    
    face_path = None
    audio_path = None
    video_path = None
    
    # Create temp directory for incoming files
    os.makedirs("data/temp", exist_ok=True)
    
    # Securely extract extension to avoid OS error 22 (invalid chars or 'C:\fakepath\' from browsers)
    face_ext = os.path.splitext(face_file.filename)[1] if face_file.filename else ".mp4"
    
    face_path = f"data/temp/{uuid.uuid4().hex}{face_ext}"
    
    # Save uploaded face file to disk
    with open(face_path, "wb") as buffer:
        shutil.copyfileobj(face_file.file, buffer)
        
    # Handle Audio: Either save uploaded file OR generate from text
    if audio_file:
        audio_ext = os.path.splitext(audio_file.filename)[1] if audio_file.filename else ".wav"
        audio_path = f"data/temp/{uuid.uuid4().hex}{audio_ext}"
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
    else:
        # Generate TTS using edge-tts
        audio_path = f"data/temp/{uuid.uuid4().hex}.wav"
        voice = "en-US-GuyNeural" if voice_gender.lower() == "male" else "en-US-JennyNeural"
        communicate = edge_tts.Communicate(script_text, voice)
        await communicate.save(audio_path)
        
    try:
        video_path = generate_lip_sync(face_path, audio_path, model)
        
        return FileResponse(
            video_path, 
            media_type="video/mp4", 
            filename="lip_sync_result.mp4",
            background=BackgroundTask(remove_file, video_path)
        )
    except Exception as e:
        remove_file(video_path)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary input files after generation
        remove_file(face_path)
        remove_file(audio_path)

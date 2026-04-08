import os
import shutil
import uuid
from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse
from typing import Annotated
from app.services.wav2lip_service import generate_lip_sync

router = APIRouter()

@router.post("/generate")
async def generate_video(
    request: Request,
    face_file: Annotated[UploadFile, File(description="Image or Video file of the face")],
    audio_file: Annotated[UploadFile, File(description="Audio file containing speech")]
):
    model = request.app.state.wav2lip_model
    
    # Create temp directory for incoming files
    os.makedirs("data/temp", exist_ok=True)
    
    # Securely extract extension to avoid OS error 22 (invalid chars or 'C:\fakepath\' from browsers)
    face_ext = os.path.splitext(face_file.filename)[1] if face_file.filename else ".mp4"
    audio_ext = os.path.splitext(audio_file.filename)[1] if audio_file.filename else ".wav"
    
    face_path = f"data/temp/{uuid.uuid4().hex}{face_ext}"
    audio_path = f"data/temp/{uuid.uuid4().hex}{audio_ext}"
    
    # Save uploaded files to disk
    with open(face_path, "wb") as buffer:
        shutil.copyfileobj(face_file.file, buffer)
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)
        
    try:
        video_path = generate_lip_sync(face_path, audio_path, model)
        return FileResponse(video_path, media_type="video/mp4", filename="lip_sync_result.mp4")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary input files after generation
        if os.path.exists(face_path): os.remove(face_path)
        if os.path.exists(audio_path): os.remove(audio_path)

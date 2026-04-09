import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from app.api.routes import router
from app.services.wav2lip_service import load_model

logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model on startup
    app.state.wav2lip_model = load_model("data/models/wav2lip_gan.pth")
    yield
    # Cleanup on shutdown
    app.state.wav2lip_model = None

app = FastAPI(title="Wav2Lip API Service", lifespan=lifespan)

# Set up allowed origins for CORS
origins = [
    "http://localhost:3000",
    os.environ.get("FRONTEND_URL")  # Your deployed frontend URL goes here
]
origins = [origin for origin in origins if origin] # Remove None if not set yet

# Configure CORS to allow the Next.js frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
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

# Configure CORS to allow the Next.js frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js development server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

import os
import uuid
import torch
import logging
from wav2lip import inference
from wav2lip.models import Wav2Lip

logger = logging.getLogger(__name__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(path: str = "data/models/wav2lip_gan.pth"):
    logger.info(f"Loading Wav2Lip model from {path} onto {device}...")
    model = Wav2Lip()
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    s = checkpoint["state_dict"]
    new_s = {k.replace('module.', ''): v for k, v in s.items()}
    model.load_state_dict(new_s)
    model = model.to(device)
    logger.info("Model loaded successfully!")
    return model.eval()

def generate_lip_sync(face_path: str, audio_path: str, model) -> str:
    os.makedirs("data/outputs", exist_ok=True)
    output_video_path = f"data/outputs/res_{uuid.uuid4().hex}.mp4"
    
    # Set the output file path in the inference module's args
    inference.args.outfile = output_video_path
    
    # Call inference with the three required parameters
    try:
        inference.main(face_path, audio_path, model)
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise

    if not os.path.exists(output_video_path):
        logger.error(f"Expected output file missing: {output_video_path}")
        raise RuntimeError(f"Inference completed but output file was not created: {output_video_path}")

    return output_video_path

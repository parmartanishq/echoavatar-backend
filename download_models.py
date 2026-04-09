import os
import urllib.request

# Safe HuggingFace mirror for the Wav2Lip GAN model
MODEL_URL = "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth"
MODEL_DIR = "data/models"
MODEL_PATH = os.path.join(MODEL_DIR, "wav2lip_gan.pth")

def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading Wav2Lip model from HuggingFace...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("✅ Model download complete!")
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
    else:
        print("✅ Model already exists. Skipping download.")

if __name__ == "__main__":
    download_model()
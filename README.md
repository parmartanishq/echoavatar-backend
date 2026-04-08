# Echo Avatar (AI Backend)

This folder contains the core backend and AI models for the Echo Avatar project. It is built in Python and utilizes PyTorch for model processing.

## 🚀 Setup & Installation

1. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables:**
   Copy the example environment file or create a `.env` file for your secrets and API keys.

## 🧠 Models

Ensure that your PyTorch model weights (`*.pth`) are placed inside the `data/models/` directory. These files are ignored by git to prevent large files from cluttering the repository. Outputs will be generated in `data/outputs/`.

## 🏃‍♂️ Running the Backend

Execute your main Python script to start the process:

```bash
uv run run.py
```

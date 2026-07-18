import torch
import torch.nn as nn
import cv2
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import gradio as gr
import os
import torchaudio
import tempfile
import subprocess

MODEL_PATH = "baseline_pvm_model.pt"
DEVICE = torch.device("cpu")
SR = 16000

MAX_FRAMES = 30
STRIDE = 10

# ---------------- MODEL ----------------
class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_proj = nn.Linear(768, 256)
        self.video_proj = nn.Linear(512, 256)

        # IMPORTANT: wrapped to match checkpoint
        self.cross_attn = nn.Module()
        self.cross_attn.attn = nn.MultiheadAttention(256, 4, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, audio_feat, video_feat):
        a = self.audio_proj(audio_feat)
        v = self.video_proj(video_feat)

        fused, _ = self.cross_attn.attn(
            a.unsqueeze(1),
            v.unsqueeze(1),
            v.unsqueeze(1)
        )

        fused = fused.squeeze(1)
        return torch.sigmoid(self.classifier(fused))


# ---------------- LOAD MODELS ----------------
print("[INFO] Loading models...")

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)
wav2vec.eval()

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = nn.Identity()
resnet = resnet.to(DEVICE)
resnet.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

model = FusionModel().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

if "model_state" in checkpoint:
    model.load_state_dict(checkpoint["model_state"])
else:
    model.load_state_dict(checkpoint)

model.eval()


# ---------------- HELPERS ----------------
def confidence_to_text(c):
    if c <= 0.20:
        return "Strong REAL"
    elif c <= 0.40:
        return "Probably REAL"
    elif c <= 0.60:
        return "Uncertain"
    elif c <= 0.80:
        return "Probably FAKE"
    else:
        return "Strong FAKE"


def confidence_reason(c):
    if c <= 0.20:
        return "Audio and facial motion patterns closely match natural human behavior."
    elif c <= 0.40:
        return "Mostly natural signals with minor inconsistencies."
    elif c <= 0.60:
        return "Model detected mixed real and synthetic features."
    elif c <= 0.80:
        return "Significant audio-visual mismatches detected."
    else:
        return "Strong phoneme-viseme mismatch and synthetic artifacts detected."


def extract_audio_feat(video_path):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    cmd = ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", str(SR), wav_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    waveform, sr = torchaudio.load(wav_path)
    os.remove(wav_path)

    waveform = waveform.squeeze(0)

    inputs = processor(
        waveform.numpy(),
        sampling_rate=SR,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        out = wav2vec(**inputs).last_hidden_state.mean(dim=1)

    return out


def extract_video_feats(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    i = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if i % STRIDE == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform(frame)
            frames.append(frame)
            if len(frames) >= MAX_FRAMES:
                break
        i += 1

    cap.release()

    if len(frames) == 0:
        raise ValueError("No frames could be read.")

    frames = torch.stack(frames).to(DEVICE)

    with torch.no_grad():
        feats = resnet(frames)

    return feats


# ---------------- PREDICT ----------------
def predict(video):
    if video is None or video == "":
        return {}, "⬆️ Please upload a video and click Submit."

    if isinstance(video, dict):
        video_path = video.get("path", None)
    else:
        video_path = video

    if video_path is None or not os.path.exists(video_path):
        return {}, "⚠️ Invalid video file."

    try:
        audio_feat = extract_audio_feat(video_path)
        video_feats = extract_video_feats(video_path)

        probs = []
        with torch.no_grad():
            for vf in video_feats:
                vf = vf.unsqueeze(0)
                p = model(audio_feat, vf).item()
                probs.append(p)

        prob = float(np.mean(probs))

        label = "FAKE" if prob >= 0.5 else "REAL"
        meaning = confidence_to_text(prob)
        reason = confidence_reason(prob)

        explanation = f"""
### Interpretation  
**{meaning}**

### Detailed Explanation  
{reason}

### Important Notes for Reliable Results
• The model performs best when a human face is clearly visible.  
• Ensure the speaker's lips and facial region are not occluded.  
• Videos with long off-screen audio may reduce accuracy.  
• Best performance is achieved with a single visible speaker.  
• Extreme camera angles or heavy compression can affect predictions.

This system analyzes **phoneme–viseme consistency** between speech audio  
and facial movements to identify potential deepfake manipulations.
"""

        return {label: prob}, explanation

    except Exception as e:
        print("RUNTIME ERROR:", e)
        return {}, "⚠️ Could not process this video. Please upload a clear talking-face video."


# ---------------- GRADIO UI ----------------
iface = gr.Interface(
    fn=predict,
    inputs=gr.Video(label="Upload a video file"),
    outputs=[
        gr.Label(label="Prediction (with confidence)"),
        gr.Markdown()
    ],
    title="Phoneme–Viseme Deepfake Detection",
    description="""
This system detects deepfake videos by analyzing the consistency between spoken audio (**phonemes**) 
and facial lip movements (**visemes**).

### Recommended Usage Guidelines
• Upload videos where a **human face is clearly visible**.  
• The person should be **speaking for most of the video duration**.  
• Avoid videos with **long off-screen narration**.  
• Best results are obtained with a **single speaker facing the camera**.  
• Heavy motion blur, occlusion, or low resolution may affect predictions.

### How it Works
The model extracts speech features using **Wav2Vec2** and facial motion features using a **CNN (ResNet18)**.  
It then evaluates their alignment using a **cross-attention fusion model** to detect inconsistencies 
commonly found in synthetic or manipulated videos.

### Project Repository  
🔗 https://github.com/Tilak-Kateghar/phoneme-viseme-deepfake-detection
""",
    theme="soft"
)

iface.launch()
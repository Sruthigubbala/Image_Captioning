"""
blip_captioner.py
Wraps Salesforce's pretrained BLIP captioning model.
Fully independent of the custom CLIP+LSTM pipeline — no training required.
"""

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

_MODEL_NAME = "Salesforce/blip-image-captioning-base"
_processor = None
_model = None


def load_blip(device="cpu"):
    """Lazy-loads BLIP once, reuses across calls."""
    global _processor, _model
    if _model is None:
        _processor = BlipProcessor.from_pretrained(_MODEL_NAME)
        _model = BlipForConditionalGeneration.from_pretrained(_MODEL_NAME).to(device)
        _model.eval()
    return _processor, _model


@torch.no_grad()
def generate_blip_caption(image: Image.Image, device="cpu") -> str:
    processor, model = load_blip(device)
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=35)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


if __name__ == "__main__":
    # Smoke test on any local image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = Image.open("data/images/1000268201_693b08cb0e.jpg").convert("RGB")  # swap for a real filename
    caption = generate_blip_caption(img, device)
    print("BLIP caption:", caption)
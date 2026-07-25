"""
blip_captioner.py
Wraps Salesforce's pretrained BLIP captioning model.
Fully independent of the custom CLIP+LSTM pipeline — no training required.

FIX: num_beams is now an actual parameter passed through to model.generate().
Previously the beam_size slider in app.py was never forwarded here, so
BLIP always ran greedy decoding (num_beams=1) no matter what the UI showed.
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
def generate_blip_caption(image: Image.Image, device="cpu",
                           num_beams: int = 1, max_new_tokens: int = 35) -> str:
    """
    num_beams=1  -> greedy decoding
    num_beams>1  -> beam search (this is what was missing before)
    """
    processor, model = load_blip(device)
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        early_stopping=(num_beams > 1),
    )
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


if __name__ == "__main__":
    # Smoke test on any local image — compares greedy vs beam=5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = Image.open("data/images/1000268201_693b08cb0e.jpg").convert("RGB")  # swap for a real filename
    print("Greedy (beam=1):", generate_blip_caption(img, device, num_beams=1))
    print("Beam=5:         ", generate_blip_caption(img, device, num_beams=5))
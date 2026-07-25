"""
app.py
Streamlit demo with a model toggle:
  - "My Custom Model" -> CLIP encoder + LSTM/Attention decoder, trained on Flickr8k (~8K images)
  - "Pretrained BLIP"  -> Salesforce BLIP, pretrained on ~14M+ image-caption pairs

FIX: beam_size from the slider is now passed to generate_blip_caption() as
num_beams, so changing it actually changes BLIP's decoding behavior.
"""

import streamlit as st
import torch
from PIL import Image

from vocabulary import Vocabulary
from dataset import get_transforms
from model import EncoderCNN, DecoderRNN
from blip_captioner import generate_blip_caption

st.set_page_config(page_title="Image Captioning", page_icon="🖼️", layout="centered")

CHECKPOINT_PATH = "checkpoints/best.pth"
VOCAB_PATH = "vocab.pkl"

# ---------------- CSS ----------------
CUSTOM_CSS = """
<style>
:root {
    --accent: #6C63FF;
    --accent-dark: #4b43d6;
    --bg-soft: #f6f5ff;
    --card-border: #e4e1ff;
}

/* Page background */
.stApp {
    background: linear-gradient(180deg, #fbfaff 0%, #f3f1ff 100%);
}

/* Hide default Streamlit chrome for a cleaner look */
#MainMenu, footer {visibility: hidden;}

/* Title */
h1 {
    font-weight: 800 !important;
    background: linear-gradient(90deg, var(--accent), #ff6fa5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding-bottom: 0.2rem;
}

/* Radio buttons -> model toggle */
div[role="radiogroup"] {
    background: #ffffff;
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 0.6rem 1rem;
    box-shadow: 0 2px 8px rgba(108, 99, 255, 0.06);
}

/* Info box */
div[data-testid="stAlertContainer"] {
    border-radius: 14px !important;
    border: 1px solid var(--card-border) !important;
    background-color: var(--bg-soft) !important;
}

/* Slider accent color */
div[data-testid="stSlider"] div[role="slider"] {
    background-color: var(--accent) !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
    background-color: var(--accent) !important;
}

/* File uploader card */
section[data-testid="stFileUploaderDropzone"] {
    background: #ffffff;
    border: 2px dashed var(--card-border);
    border-radius: 14px;
}
section[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent);
}

/* Uploaded image */
div[data-testid="stImage"] img {
    border-radius: 16px;
    box-shadow: 0 6px 20px rgba(108, 99, 255, 0.15);
}

/* Buttons */
.stButton > button {
    background: var(--accent);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.5rem 1.2rem;
    font-weight: 600;
}
.stButton > button:hover {
    background: var(--accent-dark);
    color: white;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_custom_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocabulary.load(VOCAB_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    cfg = checkpoint["config"]

    encoder = EncoderCNN().to(device)
    encoder.load_state_dict(checkpoint["encoder_state"])
    encoder.eval()

    decoder = DecoderRNN(
        vocab_size=len(vocab),
        embed_dim=cfg["embed_dim"],
        decoder_dim=cfg["decoder_dim"],
        encoder_dim=cfg["encoder_dim"],
        attention_dim=cfg["attention_dim"],
    ).to(device)
    decoder.load_state_dict(checkpoint["decoder_state"])
    decoder.eval()

    return encoder, decoder, vocab, device


def generate_custom_caption(image, encoder, decoder, vocab, device, beam_size=3):
    transform = get_transforms(train=False)
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = encoder(image_tensor)
        seq = decoder.generate(features, vocab, beam_size=beam_size, device=device)
    return vocab.decode_indices(seq)


def main():
    st.title("🖼️ Image Captioning")
    st.write("Upload an image and choose which model generates the caption.")

    model_choice = st.radio(
        "Choose a model",
        ["My Custom Model (CLIP + LSTM Attention)", "Pretrained BLIP"],
        horizontal=True,
    )

    # --- Explanatory note about the two models, shown right under the toggle ---
    if model_choice.startswith("My Custom"):
        st.info(
            "ℹ️ **About this model:** This is a CNN/CLIP encoder + LSTM decoder with "
            "attention, trained **from scratch by me** on the Flickr8k dataset "
            "(~8,000 images, mostly people and outdoor scenes). It's a great "
            "demonstration of how encoder-decoder captioning works end-to-end, "
            "but because it was trained on a small, narrow dataset, captions on "
            "images outside that domain (objects, indoor scenes, screenshots, "
            "art, etc.) may be short, generic, or inaccurate. Try the "
            "**Pretrained BLIP** option for stronger general-purpose captions."
        )
    else:
        st.info(
            "ℹ️ **About this model:** BLIP is a large model pretrained by Salesforce "
            "on millions of image-caption pairs from the web. It generalizes far "
            "better to arbitrary images than a small from-scratch model can, since "
            "it has seen a much wider variety of scenes, objects, and language "
            "during training. This model is **not trained by me** — it's used here "
            "to show the contrast between a small custom model and a large "
            "pretrained one."
        )

    beam_size = st.slider("Beam size", min_value=1, max_value=5, value=3)

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Generating caption..."):
            try:
                if model_choice.startswith("My Custom"):
                    encoder, decoder, vocab, device = load_custom_model()
                    caption = generate_custom_caption(image, encoder, decoder, vocab,
                                                        device, beam_size=beam_size)
                    label = "Custom Model Caption"
                else:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    # FIX: beam_size now actually reaches BLIP as num_beams
                    caption = generate_blip_caption(image, device=device, num_beams=beam_size)
                    label = "BLIP Caption"

                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #eafff3, #e6f4ff);
                        border: 1px solid #b9f0d1;
                        border-radius: 16px;
                        padding: 1rem 1.2rem;
                        margin-top: 0.5rem;
                        font-size: 1.05rem;
                    ">
                        <span style="color:#1a7f4e; font-weight:700;">{label}:</span>
                        <span style="color:#1f2933;"> {caption}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except FileNotFoundError:
                st.error(
                    "No trained checkpoint found for the custom model. "
                    "Train it first (see train.py) and make sure "
                    "checkpoints/best.pth and vocab.pkl exist. "
                    "BLIP mode doesn't need this — try switching models above."
                )
            except Exception as e:
                st.error(f"Something went wrong generating the caption: {e}")


if __name__ == "__main__":
    main()
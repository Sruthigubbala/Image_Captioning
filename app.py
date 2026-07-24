"""
app.py
Streamlit demo: upload an image, generate a caption, show it on screen.

Run with:
    streamlit run app.py
"""

import streamlit as st
import torch
from PIL import Image

from vocabulary import Vocabulary
from dataset import get_transforms
from model import EncoderCNN, DecoderRNN

st.set_page_config(page_title="Image Captioning", page_icon="🖼️", layout="centered")

CHECKPOINT_PATH = "checkpoints/best.pth"
VOCAB_PATH = "vocab.pkl"


@st.cache_resource
def load_model():
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


def generate_caption(image, encoder, decoder, vocab, device, beam_size=3):
    transform = get_transforms(train=False)
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = encoder(image_tensor)
        seq = decoder.generate(features, vocab, beam_size=beam_size, device=device)
    return vocab.decode_indices(seq)


def main():
    st.title("🖼️ Image Captioning")
    st.write("Upload an image and the model will generate a caption using a "
             "CNN encoder + LSTM decoder with attention.")

    beam_size = st.slider("Beam size", min_value=1, max_value=5, value=3)

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Generating caption..."):
            try:
                encoder, decoder, vocab, device = load_model()
                caption = generate_caption(image, encoder, decoder, vocab,
                                            device, beam_size=beam_size)
                st.success(f"**Caption:** {caption}")
            except FileNotFoundError:
                st.error(
                    "No trained checkpoint found. Train the model first "
                    "(see train.py) and make sure checkpoints/best.pth "
                    "and vocab.pkl exist."
                )


if __name__ == "__main__":
    main()
"""
inference.py
Load a trained checkpoint and:
  1. Generate a caption for a single image (with attention weights)
  2. Evaluate BLEU score over a validation set
"""

import torch
import matplotlib.pyplot as plt
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from vocabulary import Vocabulary
from dataset import get_transforms, load_captions_file
from model import EncoderCNN, DecoderRNN


def load_model(checkpoint_path, vocab_path, device):
    vocab = Vocabulary.load(vocab_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
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

    return encoder, decoder, vocab


@torch.no_grad()
def caption_image(image_path, encoder, decoder, vocab, device, beam_size=3, max_len=35):
    transform = get_transforms(train=False)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    features = encoder(image_tensor)  # (1, 49, 2048)
    seq = decoder.generate(features, vocab, max_len=max_len,
                            beam_size=beam_size, device=device)
    caption = vocab.decode_indices(seq)
    return caption, image


@torch.no_grad()
def evaluate_bleu(encoder, decoder, vocab, images_dir, captions_path, device,
                   max_images=200, beam_size=3):
    """Groups references per image, generates a hypothesis, computes corpus BLEU."""
    pairs = load_captions_file(captions_path)
    refs_by_image = {}
    for img_name, cap in pairs:
        refs_by_image.setdefault(img_name, []).append(Vocabulary.tokenize(cap))

    image_names = list(refs_by_image.keys())[:max_images]
    references, hypotheses = [], []
    smoothie = SmoothingFunction().method4

    for img_name in image_names:
        img_path = f"{images_dir}/{img_name}"
        try:
            hyp_caption, _ = caption_image(img_path, encoder, decoder, vocab,
                                            device, beam_size=beam_size)
        except FileNotFoundError:
            continue
        hypotheses.append(hyp_caption.split())
        references.append(refs_by_image[img_name])

    bleu4 = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
    print(f"Corpus BLEU-4 over {len(hypotheses)} images: {bleu4:.4f}")
    return bleu4


def show_attention(image, caption_words, alphas, num_pixels_side=7):
    """Plots the image with an attention heatmap overlay per generated word."""
    n_words = len(caption_words)
    cols = 5
    rows = (n_words + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if n_words > 1 else [axes]

    for i, word in enumerate(caption_words):
        ax = axes[i]
        ax.imshow(image)
        if i < len(alphas):
            alpha_map = alphas[i].reshape(num_pixels_side, num_pixels_side)
            ax.imshow(alpha_map, alpha=0.6, cmap="jet",
                      extent=(0, image.width, image.height, 0))
        ax.set_title(word)
        ax.axis("off")

    for j in range(n_words, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig("attention_visualization.png")
    print("Saved attention_visualization.png")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, vocab = load_model("checkpoints/best.pth", "vocab.pkl", device)

    caption, image = caption_image("data/images/1000268201_693b08cb0e.jpg", encoder, decoder, vocab, device)
    print("Generated caption:", caption)

    # Optional: full validation BLEU score
    # evaluate_bleu(encoder, decoder, vocab, "data/images", "data/captions.txt", device)
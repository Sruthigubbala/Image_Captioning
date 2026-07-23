"""
train.py
End-to-end training loop for the image captioning model.

Usage:
    python train.py
(edit the CONFIG section below to match your paths)
"""

import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from dataset import build_vocab_from_captions, get_dataloader
from model import EncoderCNN, DecoderRNN

# ---------------- CONFIG ----------------
CONFIG = {
    "images_dir": "data/images",
    "captions_path": "data/captions.txt",
    "vocab_path": "vocab.pkl",
    "checkpoint_dir": "checkpoints",
    "batch_size": 16,
    "max_len": 35,
    "embed_dim": 256,
    "decoder_dim": 512,
    "attention_dim": 256,
    "encoder_dim": 2048,
    "dropout": 0.5,
    "lr_decoder": 4e-4,
    "lr_encoder": 1e-5,
    "fine_tune_encoder": False,
    "num_epochs": 3,
    "grad_clip": 5.0,
    "freq_threshold": 5,
}


def train_one_epoch(encoder, decoder, dataloader, criterion,
                     optimizer, device, grad_clip):
    decoder.train()
    if CONFIG["fine_tune_encoder"]:
        encoder.train()
    else:
        encoder.eval()

    total_loss = 0.0
    for batch_idx, (images, captions) in enumerate(dataloader):
        images, captions = images.to(device), captions.to(device)

        with torch.set_grad_enabled(CONFIG["fine_tune_encoder"]):
            features = encoder(images)

        predictions, alphas = decoder(features, captions)

        targets = captions[:, 1:]  # shift right (predict next word)

        # Flatten for CrossEntropyLoss, ignore <pad>
        loss = criterion(
            predictions.reshape(-1, predictions.size(-1)),
            targets.reshape(-1),
        )

        # Doubly stochastic attention regularization (Show, Attend & Tell)
        alpha_reg = 1.0
        loss = loss + alpha_reg * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
        if CONFIG["fine_tune_encoder"]:
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"  batch {batch_idx}/{len(dataloader)}  loss={loss.item():.4f}")

    return total_loss / len(dataloader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # ---- Vocabulary ----
    if os.path.exists(CONFIG["vocab_path"]):
        from vocabulary import Vocabulary
        vocab = Vocabulary.load(CONFIG["vocab_path"])
        print(f"Loaded existing vocab ({len(vocab)} tokens)")
    else:
        vocab = build_vocab_from_captions(CONFIG["captions_path"], CONFIG["freq_threshold"])
        vocab.save(CONFIG["vocab_path"])

    # ---- Data ----
    train_loader = get_dataloader(
        CONFIG["images_dir"], CONFIG["captions_path"], vocab,
        batch_size=CONFIG["batch_size"], max_len=CONFIG["max_len"], train=True,
    )

    # ---- Model ----
    encoder = EncoderCNN(fine_tune=CONFIG["fine_tune_encoder"]).to(device)
    decoder = DecoderRNN(
        vocab_size=len(vocab),
        embed_dim=CONFIG["embed_dim"],
        decoder_dim=CONFIG["decoder_dim"],
        encoder_dim=CONFIG["encoder_dim"],
        attention_dim=CONFIG["attention_dim"],
        dropout=CONFIG["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx[vocab.PAD_TOKEN])

    params = list(decoder.parameters())
    if CONFIG["fine_tune_encoder"]:
        params += list(filter(lambda p: p.requires_grad, encoder.parameters()))

    optimizer = torch.optim.Adam([
        {"params": decoder.parameters(), "lr": CONFIG["lr_decoder"]},
    ])

    best_loss = float("inf")
    for epoch in range(1, CONFIG["num_epochs"] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['num_epochs']}")
        avg_loss = train_one_epoch(
            encoder, decoder, train_loader, criterion,
            optimizer, device, CONFIG["grad_clip"],
        )
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

        checkpoint = {
            "epoch": epoch,
            "encoder_state": encoder.state_dict(),
            "decoder_state": decoder.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": CONFIG,
        }
        torch.save(checkpoint, os.path.join(CONFIG["checkpoint_dir"], "last.pth"))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, os.path.join(CONFIG["checkpoint_dir"], "best.pth"))
            print("  -> saved new best checkpoint")


if __name__ == "__main__":
    main()
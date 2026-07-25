"""
train.py
End-to-end training loop for the image captioning model.

Usage:
    python train.py
(edit the CONFIG section below to match your paths)

FIX: added a validation loop. best.pth is now saved based on VALIDATION
loss, not training loss. Before, "best" could just be the checkpoint most
overfit to the training captions, since train loss always trends down.
"""

import os
import torch
import torch.nn as nn

from dataset import build_vocab_from_captions, get_train_val_dataloaders
from model import EncoderCNN, DecoderRNN

# ---------------- CONFIG ----------------
CONFIG = {
    "images_dir": "data/images",
    "captions_path": "data/captions.txt",
    "vocab_path": "vocab.pkl",
    "checkpoint_dir": "checkpoints",
    "batch_size": 32,
    "max_len": 35,
    "embed_dim": 256,
    "decoder_dim": 512,
    "attention_dim": 256,
    "encoder_dim": 768,
    "dropout": 0.5,
    "lr_decoder": 4e-4,
    "lr_encoder": 1e-5,
    "fine_tune_encoder": False,
    "num_epochs": 15,
    "grad_clip": 5.0,
    "freq_threshold": 5,
    "val_ratio": 0.1,          # NEW: fraction of images held out for validation
    "early_stop_patience": 4,  # NEW: stop if val loss doesn't improve for N epochs
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


@torch.no_grad()
def validate(encoder, decoder, dataloader, criterion, device):
    """NEW: runs the model on held-out data with no gradient updates."""
    encoder.eval()
    decoder.eval()

    total_loss = 0.0
    for images, captions in dataloader:
        images, captions = images.to(device), captions.to(device)
        features = encoder(images)
        predictions, alphas = decoder(features, captions)

        targets = captions[:, 1:]
        loss = criterion(
            predictions.reshape(-1, predictions.size(-1)),
            targets.reshape(-1),
        )
        loss = loss + ((1.0 - alphas.sum(dim=1)) ** 2).mean()
        total_loss += loss.item()

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

    # ---- Data ---- (NEW: train/val split by image, not by caption row)
    train_loader, val_loader = get_train_val_dataloaders(
        CONFIG["images_dir"], CONFIG["captions_path"], vocab,
        batch_size=CONFIG["batch_size"], max_len=CONFIG["max_len"],
        val_ratio=CONFIG["val_ratio"],
    )
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

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

    optimizer = torch.optim.Adam([
        {"params": decoder.parameters(), "lr": CONFIG["lr_decoder"]},
    ])

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['num_epochs']}")
        train_loss = train_one_epoch(
            encoder, decoder, train_loader, criterion,
            optimizer, device, CONFIG["grad_clip"],
        )
        val_loss = validate(encoder, decoder, val_loader, criterion, device)
        print(f"Epoch {epoch}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        checkpoint = {
            "epoch": epoch,
            "encoder_state": encoder.state_dict(),
            "decoder_state": decoder.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": CONFIG,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, os.path.join(CONFIG["checkpoint_dir"], "last.pth"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(checkpoint, os.path.join(CONFIG["checkpoint_dir"], "best.pth"))
            print("  -> saved new best checkpoint (val_loss improved)")
        else:
            epochs_without_improvement += 1
            print(f"  -> no val improvement ({epochs_without_improvement}/"
                  f"{CONFIG['early_stop_patience']})")

        if epochs_without_improvement >= CONFIG["early_stop_patience"]:
            print(f"Early stopping at epoch {epoch} — val loss hasn't "
                  f"improved in {CONFIG['early_stop_patience']} epochs.")
            break


if __name__ == "__main__":
    main()
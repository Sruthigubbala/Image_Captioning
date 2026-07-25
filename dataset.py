"""
dataset.py
PyTorch Dataset + collate_fn for Flickr8k.

Expected directory layout:
    data/
        images/                 # all .jpg files
        captions.txt            # "image_name,caption" per line (Flickr8k format)

captions.txt has 5 rows per image (one per caption).

FIX: added get_train_val_dataloaders(), which splits by unique image
filename (not by caption row) before building loaders. Splitting by row
would leak the same image into both train and val (since each image has
5 caption rows), making a "validation loss" meaningless.
"""

import os
import csv
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from vocabulary import Vocabulary

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool = True):
    """CLIP normalization stats — different from ImageNet's ResNet stats."""
    CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def load_captions_file(captions_path: str):
    """Returns list of (image_filename, caption_text) tuples."""
    pairs = []
    with open(captions_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # Handle files with or without a header row
        if header and "image" not in header[0].lower():
            pairs.append((header[0], header[1]))
        for row in reader:
            if len(row) < 2:
                continue
            pairs.append((row[0].strip(), row[1].strip()))
    return pairs


class Flickr8kDataset(Dataset):
    def __init__(self, images_dir, pairs, vocab: Vocabulary,
                 max_len: int = 35, train: bool = True):
        """
        pairs: list of (image_filename, caption_text) already filtered
               to the desired split (train or val).
        """
        self.images_dir = images_dir
        self.pairs = pairs
        self.vocab = vocab
        self.max_len = max_len
        self.transform = get_transforms(train)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        img_name, caption = self.pairs[index]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        caption_ids = self.vocab.encode_caption(caption, self.max_len)
        caption_tensor = torch.tensor(caption_ids, dtype=torch.long)

        return image, caption_tensor


def build_vocab_from_captions(captions_path: str, freq_threshold: int = 5):
    pairs = load_captions_file(captions_path)
    all_captions = [c for _, c in pairs]
    vocab = Vocabulary(freq_threshold=freq_threshold)
    vocab.build_vocabulary(all_captions)
    return vocab


def split_pairs_by_image(pairs, val_ratio: float = 0.1, seed: int = 42):
    """
    Splits (image, caption) pairs into train/val by UNIQUE IMAGE FILENAME,
    so all 5 captions of a given image stay together on the same side.
    Without this, the same image could appear in both train and val,
    which would make validation loss meaningless (data leakage).
    """
    unique_images = sorted({img for img, _ in pairs})
    rng = random.Random(seed)
    rng.shuffle(unique_images)

    n_val = max(1, int(len(unique_images) * val_ratio))
    val_images = set(unique_images[:n_val])
    train_images = set(unique_images[n_val:])

    train_pairs = [(img, cap) for img, cap in pairs if img in train_images]
    val_pairs = [(img, cap) for img, cap in pairs if img in val_images]
    return train_pairs, val_pairs


def get_dataloader(images_dir, pairs, vocab, batch_size=32,
                    max_len=35, train=True, num_workers=0):
    dataset = Flickr8kDataset(images_dir, pairs, vocab, max_len, train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )


def get_train_val_dataloaders(images_dir, captions_path, vocab, batch_size=32,
                               max_len=35, val_ratio=0.1, num_workers=0, seed=42):
    """
    Convenience wrapper used by train.py: loads captions.txt once, splits
    by image, and returns (train_loader, val_loader).
    """
    all_pairs = load_captions_file(captions_path)
    train_pairs, val_pairs = split_pairs_by_image(all_pairs, val_ratio, seed)

    train_loader = get_dataloader(images_dir, train_pairs, vocab, batch_size,
                                   max_len, train=True, num_workers=num_workers)
    val_loader = get_dataloader(images_dir, val_pairs, vocab, batch_size,
                                 max_len, train=False, num_workers=num_workers)
    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage (paths will need to match your local data layout)
    CAPTIONS_PATH = "data/captions.txt"
    IMAGES_DIR = "data/images"

    vocab = build_vocab_from_captions(CAPTIONS_PATH, freq_threshold=5)
    vocab.save("vocab.pkl")

    train_loader, val_loader = get_train_val_dataloaders(
        IMAGES_DIR, CAPTIONS_PATH, vocab, batch_size=4, val_ratio=0.1
    )
    images, captions = next(iter(train_loader))
    print("Train batches:", len(train_loader), " Val batches:", len(val_loader))
    print("Image batch shape:", images.shape)
    print("Caption batch shape:", captions.shape)
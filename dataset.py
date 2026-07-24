"""
dataset.py
PyTorch Dataset + collate_fn for Flickr8k.

Expected directory layout:
    data/
        images/                 # all .jpg files
        captions.txt            # "image_name,caption" per line (Flickr8k format)

captions.txt has 5 rows per image (one per caption).
"""

import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from vocabulary import Vocabulary

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
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
    def __init__(self, images_dir, captions_path, vocab: Vocabulary,
                 max_len: int = 35, train: bool = True):
        self.images_dir = images_dir
        self.pairs = load_captions_file(captions_path)
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


def get_dataloader(images_dir, captions_path, vocab, batch_size=32,
                    max_len=35, train=True, num_workers=0):
    dataset = Flickr8kDataset(images_dir, captions_path, vocab, max_len, train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )


if __name__ == "__main__":
    # Example usage (paths will need to match your local data layout)
    CAPTIONS_PATH = "data/captions.txt"
    IMAGES_DIR = "data/images"

    vocab = build_vocab_from_captions(CAPTIONS_PATH, freq_threshold=5)
    vocab.save("vocab.pkl")

    loader = get_dataloader(IMAGES_DIR, CAPTIONS_PATH, vocab, batch_size=4)
    images, captions = next(iter(loader))
    print("Image batch shape:", images.shape)
    print("Caption batch shape:", captions.shape)
"""
model.py
EncoderCNN  : pretrained ResNet50 (frozen backbone) -> spatial feature map
Attention   : Bahdanau (additive) attention over the CNN feature map
DecoderRNN  : LSTM that attends to image features at every timestep
"""

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import CLIPVisionModel


class EncoderCNN(nn.Module):
    """CLIP's vision encoder, frozen by default. Outputs (B, 49, 768)."""

    def __init__(self, fine_tune: bool = False,
                 clip_model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained(clip_model_name)
        self.fine_tune(fine_tune)

    def fine_tune(self, fine_tune: bool):
        for param in self.clip.parameters():
            param.requires_grad = False
        if fine_tune:
            # Unfreeze only the last transformer block
            for param in self.clip.vision_model.encoder.layers[-1].parameters():
                param.requires_grad = True

    def forward(self, images):
        """images: (B, 3, 224, 224) -> features: (B, 49, 768)"""
        outputs = self.clip(pixel_values=images)
        # last_hidden_state: (B, 50, 768) -> index 0 is the [CLS] token, drop it
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]  # (B, 49, 768)
        return patch_embeddings

class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        encoder_out: (B, num_pixels, encoder_dim)
        decoder_hidden: (B, decoder_dim)
        returns: context (B, encoder_dim), alpha (B, num_pixels)
        """
        att1 = self.encoder_att(encoder_out)                    # (B, num_pixels, attn_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)     # (B, 1, attn_dim)
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)   # (B, num_pixels)
        alpha = self.softmax(att)                                # (B, num_pixels)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (B, encoder_dim)
        return context, alpha


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, decoder_dim=512,
                 encoder_dim=768, attention_dim=256, dropout=0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # gating scalar for context
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)  # (B, encoder_dim)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, captions):
        """
        Teacher-forcing forward pass (training).
        encoder_out: (B, num_pixels, encoder_dim)
        captions: (B, max_len)  -- includes <start> ... <end> <pad>...
        returns: predictions (B, max_len-1, vocab_size), alphas (B, max_len-1, num_pixels)
        """
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)
        decode_len = captions.size(1) - 1  # predict next word at each step

        embeddings = self.embedding(captions)  # (B, max_len, embed_dim)
        h, c = self.init_hidden_state(encoder_out)

        predictions = torch.zeros(batch_size, decode_len, self.vocab_size,
                                   device=encoder_out.device)
        alphas = torch.zeros(batch_size, decode_len, num_pixels,
                              device=encoder_out.device)

        for t in range(decode_len):
            context, alpha = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))
            context = gate * context

            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            preds = self.fc(self.dropout(h))
            predictions[:, t, :] = preds
            alphas[:, t, :] = alpha

        return predictions, alphas

    def generate(self, encoder_out, vocab, max_len=35, beam_size=3, device="cpu"):
        """
        Beam search decoding for inference (single image, batch size 1).
        Returns best caption token-id list and the attention weights per step.
        """
        k = beam_size
        vocab_size = self.vocab_size

        encoder_out = encoder_out.expand(k, -1, -1)  # (k, num_pixels, encoder_dim)
        h, c = self.init_hidden_state(encoder_out)

        start_idx = vocab.word2idx[vocab.START_TOKEN]
        end_idx = vocab.word2idx[vocab.END_TOKEN]

        seqs = torch.full((k, 1), start_idx, dtype=torch.long, device=device)
        top_scores = torch.zeros(k, 1, device=device)
        complete_seqs, complete_scores = [], []
        alphas_list = [[] for _ in range(k)]

        step = 1
        while True:
            embeddings = self.embedding(seqs[:, -1])  # (k, embed_dim)
            context, alpha = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))
            context = gate * context

            lstm_input = torch.cat([embeddings, context], dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            scores = torch.log_softmax(self.fc(h), dim=1)  # (k, vocab_size)

            scores = top_scores.expand_as(scores) + scores
            if step == 1:
                top_scores, top_idx = scores[0].topk(k, 0)
            else:
                top_scores, top_idx = scores.view(-1).topk(k, 0)

            prev_word_idx = top_idx // vocab_size
            next_word_idx = top_idx % vocab_size

            seqs = torch.cat([seqs[prev_word_idx], next_word_idx.unsqueeze(1)], dim=1)
            top_scores = top_scores.unsqueeze(1)

            incomplete = [i for i, w in enumerate(next_word_idx) if w != end_idx]
            complete = [i for i, w in enumerate(next_word_idx) if w == end_idx]

            for i in complete:
                complete_seqs.append(seqs[i].tolist())
                complete_scores.append(top_scores[i].item())

            k -= len(complete)
            if k == 0 or step >= max_len:
                break

            seqs = seqs[incomplete]
            h = h[prev_word_idx[incomplete]]
            c = c[prev_word_idx[incomplete]]
            encoder_out = encoder_out[prev_word_idx[incomplete]]
            top_scores = top_scores[incomplete]
            step += 1

        if not complete_seqs:
            complete_seqs = seqs.tolist()
            complete_scores = top_scores.squeeze(1).tolist()

        best_idx = complete_scores.index(max(complete_scores))
        return complete_seqs[best_idx]


if __name__ == "__main__":
    # Smoke test with random tensors
    B, vocab_size, max_len = 2, 500, 20
    encoder = EncoderCNN()
    decoder = DecoderRNN(vocab_size=vocab_size)

    dummy_images = torch.randn(B, 3, 224, 224)
    dummy_captions = torch.randint(0, vocab_size, (B, max_len))

    features = encoder(dummy_images)
    preds, alphas = decoder(features, dummy_captions)
    print("Encoder output:", features.shape)
    print("Predictions:", preds.shape)
    print("Alphas:", alphas.shape)
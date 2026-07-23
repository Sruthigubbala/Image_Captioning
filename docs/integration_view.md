flowchart TD
    A["Flickr8k_Dataset<br/>(Images/ + captions.txt)"] --> B["vocabulary.py<br/>build_vocab(), encode(), decode()"]
    A --> C["Custom Dataset class<br/>(image + tokenized caption pairs)"]
    B --> C
    C --> D["DataLoader<br/>(batches: image tensors + padded caption sequences)"]
    D --> E["Encoder<br/>ResNet50 / EfficientNet (pretrained, frozen or fine-tuned)"]
    D --> F["Decoder<br/>LSTM / Transformer + Bahdanau Attention"]
    E -->|"feature vector / feature map"| F
    F --> G["Training Loop<br/>(cross-entropy loss, teacher forcing)"]
    G --> H["Saved Checkpoint<br/>(model weights + vocabulary object)"]
    F --> I["Inference Pipeline<br/>(generate caption word-by-word)"]
    H --> I
    I --> J["BLEU Evaluation<br/>(compare generated vs reference captions)"]
    I --> K["Streamlit App<br/>(upload image → caption + attention heatmap overlay)"]
    F -->|"attention weights per timestep"| K

# Integration View

## Purpose

This section describes how the independently developed modules of the Image
Captioning system — vocabulary handling, data loading, encoder, decoder, and
the downstream evaluation and deployment layers — are integrated into a single
end-to-end pipeline. It focuses on the interfaces between modules rather than
the internal logic of each module, which is covered separately in the
component-level design.

## Data Flow

Raw images and their corresponding captions from the Flickr8k dataset form the
input to the pipeline. `vocabulary.py` processes the caption text first,
building a token vocabulary from the training captions and exposing
`encode()` and `decode()` functions that convert between raw text and integer
token sequences.

A custom Dataset class combines each image with its encoded caption sequence,
producing (image, caption) pairs. The DataLoader then batches these pairs,
padding caption sequences to a uniform length within each batch so they can be
processed as fixed-size tensors.

Each batch of images is passed through the Encoder (a pretrained ResNet50 or
EfficientNet backbone), which outputs a feature vector (or feature map, if
spatial features are retained for attention) summarizing the visual content of
each image. This encoded representation is passed to the Decoder — an LSTM or
lightweight Transformer equipped with Bahdanau attention — which generates the
caption one token at a time, attending back to relevant regions of the encoder
output at each decoding step.

During training, the decoder's predicted token sequence is compared against
the ground-truth encoded caption using cross-entropy loss, with teacher
forcing used to stabilize early training. Once training converges, the model
weights and the vocabulary object used during training are saved together as
a checkpoint, since both are required to reproduce consistent behavior at
inference time.

At inference, the same checkpoint and vocabulary object are loaded by two
downstream consumers: the BLEU evaluation script, which generates captions for
a held-out test set and scores them against reference captions, and the
Streamlit application, which accepts a user-uploaded image, generates a
caption via the same encoder-decoder pipeline, and overlays the decoder's
attention weights on the image to visualize which regions informed each
predicted word.

## Interface Contracts

- **Vocabulary size ↔ Decoder embedding layer:** The decoder's embedding and
  output layers are sized according to the vocabulary at training time. If the
  vocabulary is rebuilt (e.g., a different frequency threshold or additional
  data), the model must be retrained — a saved checkpoint cannot be reloaded
  against a mismatched vocabulary size.
- **Encoder output shape ↔ Decoder input shape:** The decoder expects a fixed
  feature dimensionality from the encoder (e.g., a pooled feature vector of
  shape `(batch, 2048)` for ResNet50, or a spatial feature map of shape
  `(batch, 49, 2048)` if attention operates over spatial regions). Any change
  to the encoder backbone must be matched by an equivalent change to the
  decoder's input projection layer.
- **Checkpoint ↔ Vocabulary pairing:** Model weights and the vocabulary object
  must always be loaded together as a matched pair. Loading a checkpoint with
  a different vocabulary object than the one used during training will silently
  produce incorrect or degenerate captions rather than raising an error.

## Integration Risks / Limitations

- Rebuilding the vocabulary without retraining the decoder is a common source
  of silent failure — the model will still run but produce degraded or
  meaningless output.
- A high proportion of `<unk>` tokens in decoded output typically indicates
  either a vocabulary built from too little data or too high a frequency
  threshold, rather than a genuine decoder failure — this should be checked
  before assuming a training bug.
- The Streamlit deployment layer depends on both the trained checkpoint and
  the exact vocabulary object from training; deploying with a mismatched or
  regenerated vocabulary file will misalign token indices and produce
  incorrect captions even if the underlying model weights are correct.


# Neural Machine Translation (English → German)

This project implements a **Neural Machine Translation (NMT) system** for translating English sentences into German. It combines a **sequence-to-sequence model with LSTMs** and an **adversarial Leaky GAN setup** to improve translation fluency and robustness beyond standard encoder-decoder approaches.

## 🔑 Features

* **Encoder–Decoder with Attention**: Captures long-term dependencies for sentence-level translation.
* **Leaky GAN Integration**: Introduces an adversarial discriminator to encourage fluent and natural German outputs.
* **BLEU Evaluation**: Uses BLEU scores to evaluate translation quality.
* **Batch Processing**: Efficient training with mini-batches on GPU.
* **Inference Pipeline**: Supports beam search decoding for better translations.

## 🏗️ Architecture

1. **Encoder (LSTM)**

   * Tokenized English sentences converted to embeddings.
   * Multi-layer LSTMs capture contextual representations.

2. **Decoder (LSTM + Attention)**

   * Generates German translations one token at a time.
   * Attention mechanism aligns decoder outputs with relevant encoder states.

3. **Leaky GAN Component**

   * **Generator**: The Seq2Seq model acts as the generator.
   * **Discriminator**: A classifier distinguishes between human and model-generated German translations.
   * **LeakyReLU activations** used in the discriminator for stable training.

4. **Loss Function**

   * Translation loss: Cross-entropy between predicted and target tokens.
   * GAN loss: Binary cross-entropy to fool the discriminator.
   * Final loss = λ \* translation loss + (1-λ) \* adversarial loss.
model.translate(sentence))
```

**Output:**

```
"Das Wetter ist heute schön."
```

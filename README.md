# English-Chinese Seq2Seq Translator

This is a simple sequence-to-sequence (Seq2Seq) neural machine translation (NMT) project using PyTorch. It translates **Chinese ↔ English** conversational sentences, using a GRU-based encoder-decoder architecture.

---

## Project Structure

```
.
├── main.py          # Entry point: trains model and runs interactive translation
├── model.py         # Defines Encoder, Decoder, and Seq2Seq model
├── data.py          # Loads and processes dataset, builds vocabulary
├── train.py         # Training loop per epoch
├── translate.py     # Function to translate a single sentence
```

---

## Dependencies

Make sure the following packages are installed:

```bash
pip install torch datasets
```
You just need to run the following command:
```bash
pip install -r requirements.txt
```
---

## Dataset

This project uses the [OpenSubtitles-TW-Corpus (en-zh_tw)](https://huggingface.co/datasets/Heng666/OpenSubtitles-TW-Corpus) from Hugging Face.

- **input**: English sentence
- **output**: Chinese translation
- **instruction**: Variant of task instruction (e.g. "Please translate to Chinese")

---

## Usage

### 1. **Train the Model**
Run the following command:

```bash
python main.py
```

This will:
- Load and preprocess the dataset
- Build vocabularies for source (zh) and target (en)
- Train a Seq2Seq model using GRU
- Save and run an interactive translation prompt after training

---

### 2. **Interactive Translation**

After training, you’ll be prompted to enter a Chinese sentence:

```bash
請輸入中文句子(輸入 exit 離開):
中文：你好嗎？
英文翻譯： how are you ?
```

Type `exit` to quit.

---

## Model Architecture

Implemented in `model.py`:

### Encoder
- GRU-based
- Embedding layer → GRU → Final hidden state

### Decoder
- GRU-based
- Embedding layer → GRU → Linear → Prediction

### Seq2Seq
- Takes source and target sequences
- Uses teacher forcing during training
- Outputs logits for each target token

---

## Training Details

Defined in `train.py`:
- Loss function: `CrossEntropyLoss` (ignores `<pad>`)
- Optimizer: `Adam`
- Teacher forcing ratio: 0.5
- Device-aware: uses GPU if available

---

## Translation Logic

Defined in `translate.py`:
- Tokenizes the Chinese input
- Encodes with the Encoder
- Decodes step-by-step using greedy decoding (argmax)
- Stops when `<eos>` is generated

---


## Author

Built with Albert Huang using PyTorch for educational and research purposes.

---
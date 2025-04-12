# Main entry point to train the model and perform translation

import torch
import torch.nn as nn
import torch.optim as optim

from data import load_data_from_huggingface
from model import Encoder, Decoder, Seq2Seq
from train import train
from translate import translate_sentence

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 4
MAX_SAMPLES = 100000 # Can be modify by user
N_EPOCHS = 20
EMB_DIM = 256
HID_DIM = 512

# Load data and build vocabularies
train_iterator, valid_iterator, SRC, TRG = load_data_from_huggingface(
    batch_size=BATCH_SIZE, device=device, max_samples=MAX_SAMPLES
)

# Define input/output dimensions
INPUT_DIM = len(SRC)
OUTPUT_DIM = len(TRG)

# Initialize model, optimizer, and loss function
enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM)
dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM)
model = Seq2Seq(enc, dec).to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=TRG.stoi['<pad>'])

# Training loop
print("Model Training...")
for epoch in range(N_EPOCHS):
    loss = train(model, train_iterator, optimizer, criterion)
    print(f"Epoch {epoch+1}/{N_EPOCHS} | Loss: {loss:.3f}")

print("Training completed")

# Interactive translation loop
print("\n請輸入中文句子(輸入 exit 離開): ")
while True:
    s = input("中文：")
    if s.lower() == 'exit':
        break
    result = translate_sentence(s, model, SRC, TRG, device)
    print("英文翻譯：", result)

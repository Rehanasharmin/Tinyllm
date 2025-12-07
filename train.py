# train.py
import torch
import torch.nn.functional as F
from model import TinyLLM
import time

# Load data
text = open('dialogue.txt').read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
data = torch.tensor(encode(text), dtype=torch.long)

# Config
BATCH, SEQ_LEN, EPOCHS = 32, 128, 3
device = 'cpu'

# Model
model = TinyLLM(len(chars), dim=128, n_layers=3, n_heads=4, max_len=SEQ_LEN).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
print(f"Vocabulary size: {len(chars)}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Dataset: {len(data):,} characters")

# Train
start = time.time()
for epoch in range(EPOCHS):
    total_loss = 0
    steps = 0
    for i in range(0, len(data) - SEQ_LEN - 1, BATCH * SEQ_LEN):
        idxs = [i + j * SEQ_LEN for j in range(BATCH) if i + j * SEQ_LEN + SEQ_LEN < len(data)]
        if len(idxs) < 2: continue
        x = torch.stack([data[j:j+SEQ_LEN] for j in idxs]).to(device)
        y = torch.stack([data[j+1:j+SEQ_LEN+1] for j in idxs]).to(device)
        loss = F.cross_entropy(model(x).view(-1, len(chars)), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        steps += 1
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/steps:.4f}")

elapsed = time.time() - start
print(f"\nTraining completed in {elapsed:.1f} seconds")
torch.save({'model': model.state_dict(), 'chars': chars}, 'tiny_llm.pt')
print("Saved tiny_llm.pt")

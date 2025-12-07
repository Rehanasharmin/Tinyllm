import torch
import torch.nn.functional as F
from model import TinyLLM
import time

text = open('dialogue.txt').read()[:100000]
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

BATCH, SEQ_LEN, EPOCHS = 16, 64, 2

model = TinyLLM(len(chars), dim=96, n_layers=2, n_heads=4, max_len=SEQ_LEN)
opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

start = time.time()
for epoch in range(EPOCHS):
    losses = []
    for i in range(0, len(data) - SEQ_LEN - 1, BATCH * SEQ_LEN):
        idxs = [i + j * SEQ_LEN for j in range(BATCH) if i + j * SEQ_LEN + SEQ_LEN < len(data)]
        if len(idxs) < 2: continue
        x = torch.stack([data[j:j+SEQ_LEN] for j in idxs])
        y = torch.stack([data[j+1:j+SEQ_LEN+1] for j in idxs])
        loss = F.cross_entropy(model(x).view(-1, len(chars)), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    print(f"Epoch {epoch+1}, Loss: {sum(losses)/len(losses):.4f}")

print(f"Done in {time.time()-start:.1f}s")
torch.save({'model': model.state_dict(), 'chars': chars}, 'tiny_llm.pt')
print("Saved!")

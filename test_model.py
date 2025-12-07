import os, sys
os.chdir('/workspace/tiny_llm')
sys.path.insert(0, '/workspace/tiny_llm')
import torch
from model import TinyLLM

ckpt = torch.load('tiny_llm.pt', weights_only=False)
chars = ckpt['chars']
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

model = TinyLLM(len(chars), 128, 3, 4, 128)
model.load_state_dict(ckpt['model'])
model.eval()

def generate(prompt, max_new=100):
    ids = [stoi.get(c, 0) for c in prompt]
    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor([ids[-128:]])
            logits = model(x)[0, -1] / 0.7
            probs = torch.softmax(logits, -1)
            nxt = torch.multinomial(probs, 1).item()
            ids.append(nxt)
            if ''.join(itos[i] for i in ids).endswith('<|end|>'): break
    return ''.join(itos[i] for i in ids)

tests = ["Hello!", "What is AI?", "Tell me a joke"]
for t in tests:
    prompt = f"<|user|>{t}<|bot|>"
    out = generate(prompt)
    resp = out.split('<|bot|>')[-1].replace('<|end|>', '').strip()
    print(f"User: {t}\nBot: {resp}\n")

print(f"Model size: {sum(p.numel() for p in model.parameters()):,} params")
import os
print(f"Model file: {os.path.getsize('tiny_llm.pt')/1024:.1f} KB")

# chat.py
import torch
from model import TinyLLM

# Load model
ckpt = torch.load('tiny_llm.pt', weights_only=False)
chars = ckpt['chars']
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

model = TinyLLM(len(chars), 128, 3, 4, 128)
model.load_state_dict(ckpt['model'])
model.eval()

def generate(prompt, max_new=150, temperature=0.8):
    ids = [stoi.get(c, 0) for c in prompt]
    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor([ids[-128:]])
            logits = model(x)[0, -1] / temperature
            probs = torch.softmax(logits, -1)
            nxt = torch.multinomial(probs, 1).item()
            ids.append(nxt)
            text = ''.join(itos[i] for i in ids)
            if text.endswith('<|end|>'):
                break
    return ''.join(itos[i] for i in ids)

print("=" * 50)
print("TinyLLM Chat (type 'quit' to exit)")
print("=" * 50 + "\n")

while True:
    try:
        user = input("You: ").strip()
        if user.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if not user:
            continue
        prompt = f"<|user|>{user}<|bot|>"
        out = generate(prompt)
        response = out.split('<|bot|>')[-1].replace('<|end|>', '').strip()
        print(f"Bot: {response}\n")
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break

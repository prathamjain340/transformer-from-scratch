import torch
import torch.nn.functional as F

# read in all words
words = open("names.txt", "r").read().splitlines()

# build vocabulary of characters and mappings to/from integers
char = sorted(set(''.join(words)))
stoi = {s:i+1 for i, s in enumerate(char)}
stoi["."] = 0
itos = {i:s for s, i  in stoi.items()}

# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one
X, Y = [], []
for w in words:
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)

C = torch.randn((27, 10))
W1 = torch.randn(30, 200)
b1 = torch.randn(200)
W2 = torch.randn(200, 27)
b2 = torch.randn(27)
parameters = [C, W1, b1, W2, b2]
parametersnum = sum(p.nelement() for p in parameters)
print(f'Num of parameters = {parametersnum}')

for p in parameters:
    p.requires_grad = True

epochs = 100000
for i in range(epochs):
    # mini batch construct
    ix = torch.randint(0, X.shape[0], (32,))

    # forward pass
    emb = C[X[ix]] # (32, 3, 2)
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
    logits = h @ W2 + b2 # (32, 27)
    loss = F.cross_entropy(logits, Y[ix])

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < epochs/2 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

print(f'loss: {loss.item()}')

for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim = 1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break

    print(''.join(itos[i] for i in out))

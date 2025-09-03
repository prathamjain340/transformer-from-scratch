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
    # print(w)
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        # print(''.join(itos[i] for i in context), '----->', itos[ix])
        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)

C = torch.randn((27, 2))
W1 = torch.randn(6, 100)
b1 = torch.randn(100)
W2 = torch.randn(100, 27)
b2 = torch.randn(27)
parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True

# lre = torch.linspace(-3, 0, 1000)
# lrs = 10**lre

# lri = []
# lossi = []

for i in range(10000):
    # mini batch construct
    ix = torch.randint(0, X.shape[0], (32,))

    # forward pass
    emb = C[X[ix]] # (32, 3, 2)
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (32, 100)
    logits = h @ W2 + b2 # (32, 27)
    # counts = logits.exp()
    # probs = counts / counts.sum(1, keepdims=True)
    # loss = -probs[torch.arange(32), Y].log().mean()
    loss = F.cross_entropy(logits, Y[ix])

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    # lr = lrs[i]
    lr = 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    # lri.append(lre[i])
    # lossi.append(loss.item())

print(f'loss: {loss.item()}')
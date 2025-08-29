# c:43.5 d:128
import torch
import torch.nn.functional as F
words = open("names.txt", "r").read().splitlines()

# N = torch.zeros((27, 27))
char = sorted(set(''.join(words)))
stoi = {s:i+1 for i, s in enumerate(char)}
stoi["."] = 0

itos = {i:s for s, i  in stoi.items()}

# before: to make count matrix
# for w in words:
#     chs = "." + w + "."
#     for ch1, ch2 in zip(chs, chs[1:]):
#         ix1 = stoi[ch1]
#         ix2 = stoi[ch2]
#         N[ix1, ix2] += 1

# BIGRAM WITH NEAURAL NETWORK

# create the training set of bigram (x,y)

# now: to make count matrix
xs, ys = [], []
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

# randomly initialize 27 neurons;' weights, ,each neuron receives 27 inputs
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

# epochs over the NN
# for i in range(100):
#     # Forward pass
#     xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
#     logits = xenc @ W # predict log-counts
#     # softmax
#     counts = logits.exp()# counts, equivalent to N
#     probs = counts / counts.sum(1, keepdims=True) # probabilities for the next character
#     loss = -probs[torch.arange(num), ys].log().mean()

#     # Backward pass
#     W.grad = None # set gradient to zero
#     loss.backward()
    
#     # Update
#     W.data += -50 * W.grad
#     print(loss.item())

for i in range(5):
    out = []
    ix = 0
    while True:
        # before
        # p = P[ix]

        #now
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True)

        ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print("".join(out))
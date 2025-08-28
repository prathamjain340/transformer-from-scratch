import numpy as np
words = open("names.txt", "r").read().splitlines()

N = np.zeros((28, 28), dtype=np.float32)
char = sorted(set(''.join(words)))
stoi = {s:i+1 for i, s in enumerate(char)}
stoi["."] = 0

# itos = {i:s for s, i  in stoi.items()}

# for w in words:
#     chs = "." + w + "."
#     for ch1, ch2 in zip(chs, chs[1:]):
#         ix1 = stoi[ch1]
#         ix2 = stoi[ch2]
#         N[ix1, ix2] += 1

# P = np.float32(N)
# P /= P.sum(1, keepdims=True)

# for i in range(5):
#     ix = 0
#     out = []
#     while True:
#         p = P[ix]
#         ix = np.random.choice(len(p), size=1, replace=True, p=p).item()
#         out.append(itos[ix])
#         if ix == 0:
#             break
#     # print("".join(out))

# log_likelihood = 0.0
# n = 0

# for w in words:
#     chs = "." + w + "."
#     for ch1, ch2 in zip(chs, chs[1:]):
#         ix1 = stoi[ch1]
#         ix2 = stoi[ch2]
#         prob = P[ix1, ix2]
#         log_prob = np.log(prob)
#         log_likelihood += log_prob
#         nll = -log_likelihood
#         n += 1
#         # print(f'{ch1}{ch2}: {prob=:.2f} {log_prob=:.4f}')

# print(f'{log_likelihood:.4f} {nll=}')
# loss = nll / n
# print(f'{loss=}')

# BIGRAM WITH NEAURAL NETWORK

# create the training set of bigram (x,y)

xs, ys = [], []
for w in words[:1]:
    chs = "." + w + "."
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

# xs = np.array(xs)
# ys = np.array(ys)
print(xs, ys)
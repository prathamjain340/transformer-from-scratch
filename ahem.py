import numpy as np
import re
from collections import Counter
from datasets import load_dataset
import os

class Tokenizer:
    def __init__(self, vocab_size=1000, unk_token="<UNK>", pad_token="<PAD>"):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        # self.word2id = {}
        # self.id2word = {}
        # self.pad_id = None
        # self.unk_id = None

    def tokenize(self, text):
        # Convert to lowercase and split on words (removes punctuation)
        return re.findall(r"[a-z0-9]+", text.lower())
    
    def fit(self, texts):
        # Tokenize each sentence
        all_tokens = [tok for text in texts for tok in self.tokenize(text)]
        freq = Counter(all_tokens)

        # Sort by frequency (high to low), then alphabetically
        most_common = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        most_common = most_common[:self.vocab_size - 2]  # save space for UNK, PAD

        # Assign IDs
        self.id2word = [self.pad_token, self.unk_token] + [w for w, _ in most_common]
        self.word2id = {w: i for i, w in enumerate(self.id2word)}

        self.pad_id = self.word2id[self.pad_token]
        self.unk_id = self.word2id[self.unk_token]

    def encode(self, text):
        return [self.word2id.get(tok, self.unk_id) for tok in self.tokenize(text)]

    def encode_batch(self, texts, max_len = None):
        encoded = [self.encode(text) for text in texts]
        return self.pad_sequences(encoded, max_len)
    
    # def decode(self, ids):
    #     return [self.id2word[i] if 0 <= i < len(self.id2word) else self.unk_token for i in ids]

    def pad_sequences(self, seqs, max_len):
        # if max_len is None:
        #     max_len = max(len(s) for s in seqs)
        padded, mask = [], []

        for seq in seqs:
            if len(seq) < max_len:
                padded_seq = seq + [self.pad_id] * (max_len - len(seq))
                mask_seq = [1] * len(seq) + [0] * (max_len - len(seq))
            else:
                padded_seq = seq[:max_len]
                mask_seq = [1] * max_len
            padded.append(padded_seq)
            mask.append(mask_seq)
            # pad_len = max_len - len(seq)
            # padded_seq = seq + [self.pad_id] * pad_len
            # mask_seq = [1] * len(seq) + [0] * pad_len
            # padded.append(padded_seq)
            # mask.append(mask_seq)

        return np.array(padded, dtype=np.int32), np.array(mask, dtype=np.float32)

# Embedding Layer
class Embedding:
    def __init__(self, vocab_size, embedding_dim, pad_id = None):
        self.embedding = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.pad_id = pad_id
        if self.pad_id is not None:
            self.embedding[self.pad_id] = 0.0

    def forward(self, x):
        self.input = x
        return self.embedding[x] # shape:(batch_size, seq_len, embedding_dim)
    
    def backward(self, grad_output):
        self.d_embedding = np.zeros_like(self.embedding)
        B, T = self.input.shape
        for i in range(B):
            for j in range(T):
                token_id = self.input[i, j]
                if self.pad_id is not None and token_id == self.pad_id:
                    continue
                self.d_embedding[token_id] += grad_output[i, j]
        return None
    
    def update(self, learning_rate):
        if self.pad_id is not None:
            pad_row = self.embedding[self.pad_id].copy()
        self.embedding -= learning_rate * self.d_embedding
        if self.pad_id is not None:
            self.embedding[self.pad_id] = 0.0

class PositionalEmbedding:
    def __init__(self, max_len, embedding_dim):
        self.embedding = np.random.randn(max_len, embedding_dim) * 0.01
        # self.embedding = np.zeros((max_len, embedding_dim))

    def forward(self, x):
        self.input_shape = x.shape
        self.batch_size, self.seq_len, self.embedding_dim = x.shape
        self.pos_embed = self.embedding[:self.seq_len]
        self.pos_embed_batched = np.tile(self.pos_embed, (self.batch_size, 1, 1))
        return x + self.pos_embed_batched

    def backward(self, grad_output):
        # Gradient w.r.t the positional embedding is just the sum over batches
        self.d_embedding = np.zeros_like(self.embedding)
        for i in range(self.seq_len):
            self.d_embedding[i] = grad_output[:, i, :].sum(axis = 0)
        return grad_output
    
    def update(self, learning_rate):
        self.embedding[:self.seq_len] -= learning_rate * self.d_embedding[:self.seq_len]

# Mean Pooling Layer
class MaskedMeanPooling:
    def forward(self, x, mask):
        self.input = x
        self.mask = mask.astype(np.float32)  # Ensure float for math ops
        m = self.mask[:, :, None]  # (B, T, 1) for broadcasting
        summed = (x * m).sum(axis =1 )  # (B, D)
        lengths = m.sum(axis = 1)
        self.lengths = lengths
        return summed / (lengths + 1e-8)  # (B, D)

    def backward(self, grad_output):
        m = self.mask[:, :, None]  # (B, T, 1)
        scale = (grad_output / (self.lengths + 1e-8))[:, None, :]  # (B, 1, D) broadcasted over T

        return scale * m  # (B, T, D)

# Linear Layer
class Linear:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.1
        self.b = np.zeros((1, output_dim))

    def forward(self, x):
        self.input = x
        return np.dot(x, self.W) + self.b
    
    def backward(self, grad_output):
        self.dW = np.dot(self.input.T, grad_output)
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        return np.dot(grad_output, self.W.T)
    
    def update(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

# Activation Layers
class ReLU:
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        grad = grad_output.copy()
        grad[self.input <= 0] = 0
        return grad

class Sigmoid:
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)
    
class TanH:
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output
    
    def backward(self, grad_output):
        return grad_output * (1 - self.output ** 2)
    
# Activation Map
activation_map = {
    'relu' : ReLU,
    'sigmoid' : Sigmoid,
    'tanh' : TanH,
    'none' : None
}

# MODEL

# Neural Network Class
class NeuralNetwork:
    def __init__(self, layer_config):
        # Create Layers
        self.layers = []
        for input_dim, output_dim, activation in layer_config:
            layer = Linear(input_dim, output_dim)
            act_fn = activation_map[activation]() if activation != 'none' else None
            self.layers.append((layer, act_fn))

    def forward(self, x):
        self.inputs = []
        out = x
        for layer, activation in self.layers:
            out = layer.forward(out)
            if activation:
                out = activation.forward(out)
        return out
    
    def backward(self, grad):
        for layer, activation in reversed(self.layers):
            if activation:
                grad = activation.backward(grad)
            grad = layer.backward(grad)
        return grad
    
    def compute_loss(self, logits, labels):
        self.probs = softmax(logits)
        loss = cross_entropy_batch(self.probs, labels)
        return loss, self.probs
    
    def update(self, learning_rate):
        for layer, _ in self.layers:
                layer.update(learning_rate)

    def predict(self, x):
        logits = self.forward(x)
        return np.argmax(logits, axis = 1)


def softmax(logits):
    logits_stable = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(logits_stable)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs

def cross_entropy_batch(probs, labels):
    eps = 1e-10
    correct_probs = probs[np.arange(len(labels)), labels]
    loss = -np.log(correct_probs + eps)
    return np.mean(loss)

def gradient_loss(probs, labels):
    batch_size = probs.shape[0]
    grad = probs.copy()
    grad[np.arange(batch_size), labels] -= 1    #gradient of loss w.r.t last linear layer
    return grad / batch_size

def accuracy(predictions, labels):
    return np.mean(predictions == labels)

def visualize_predictions(model, embed, pos_embed, pooling, tokenizer, texts, labels, id2label, max_len=20, num_samples=5):
    X_ids, mask = tokenizer.encode_batch(texts, max_len)
    x_embed = embed.forward(X_ids)
    x_embed += pos_embed.forward(x_embed)
    x_pooled = pooling.forward(x_embed, mask)
    logits = model.forward(x_pooled)
    probs = softmax(logits)
    preds = np.argmax(probs, axis = 1)

    for i in range(num_samples):
        print("-" * 80)
        print(f"Sentence    : {texts[i]}")
        print(f"True Label    : {id2label[labels[i]]}")
        print(f"Predicted Label    : {id2label[preds[i]]}")
        conf = {id2label[j]: round(float(probs[i][j]), 4) for j in range(len(id2label))}
        print(f"Confidence      :{conf}")

# Try our own custom sentence
def custom_text_pred(model, embedding, pos_embed, pooling, tok, custom_text, id2label):
    # Tokenize
    X_ids, mask = tok.encode_batch([custom_text], max_len=20)

    # Embedding + Positional Embedding
    x_embed = embedding.forward(X_ids)
    x_embed += pos_embed.forward(x_embed)

    # Mean Pooling
    x_pooled = pooling.forward(x_embed, mask)

    # Predict
    logits = model.forward(x_pooled)
    probs = softmax(logits)

    # Get predicted label
    pred = np.argmax(probs, axis=1)[0]
    pred_label = id2label[pred]

    # Show prediction
    print(f"Input Sentence  : {custom_text}")
    print(f"Predicted Label : {pred_label}")
    print("Confidence      :", {id2label[i]: round(float(p), 4) for i, p in enumerate(probs[0])})

# Main

dataset = load_dataset("ag_news")
train_data = dataset["train"].select(range(5000))
texts = [item["text"] for item in train_data]
labels = [item["label"] for item in train_data]

# Fit tokenizer
tok = Tokenizer()
tok.fit(texts)

# Encode + pad to a fixed length
X_ids, mask = tok.encode_batch(texts, max_len = 20)
labels = np.array(labels)

# Build model pieces using existing classes
vocab_size = len(tok.id2word)
embedding_dim = 32
hidden_dim = 64
num_classes = 4
max_len = X_ids.shape[1]

# Embedding + Pooling setup
embedding = Embedding(vocab_size, embedding_dim, pad_id=tok.pad_id)
pos_embed = PositionalEmbedding(max_len=max_len, embedding_dim=embedding_dim)
pooling = MaskedMeanPooling()

# Neural Network layer config
layer_config = [
    (embedding_dim, hidden_dim, 'relu'),
    (hidden_dim, num_classes, 'none')
]

model = NeuralNetwork(layer_config)

X_ids_orig = X_ids.copy()
mask_orig = mask.copy()
labels_orig = labels.copy()

# Training Configuration
perm = np.random.permutation(len(X_ids))
X_ids = X_ids[perm]; y = labels[perm]

epochs = 200
learning_rate = 0.1
batch_size = 2

if os.path.exists("model_checkpoint.npz"):
    data = np.load("model_checkpoint.npz")
    embedding.embedding = data["embed_weights"]
    pos_embed.embedding = data["pos_weights"]

    for i, (layer, _) in enumerate(model.layers):
        layer.W = data[f"W{i}"]
        layer.b = data[f"b{i}"]
    print("Model loaded from checkpoint.")
else:
    print("Training model from scratch.")
    for epoch in range(epochs):
        # shuffle every epoch
        indices = np.random.permutation(len(X_ids))
        X_ids = X_ids[indices]
        mask = mask[indices]
        y = y[indices]

        # mini-batches
        for i in range(0, len(X_ids), batch_size):
            xb = X_ids[i:i+batch_size]           # (B, T)
            mb = mask[i:i+batch_size]
            yb = y[i:i+batch_size]               # (B,)
            # mask = (xb != tok.pad_id).astype(np.float32)

            x_embed = embedding.forward(xb)      # (B, T, D)
            x_embed = pos_embed.forward(x_embed)
            x_pooled = pooling.forward(x_embed, mb)  # (B, D)
            # x_pooled = pooling.forward(x_embed, mask)  # (B, D)
            logits = model.forward(x_pooled)     # (B, C)
            probs = softmax(logits)              # (B, C)
            loss = cross_entropy_batch(probs, yb)

            grad = gradient_loss(probs, yb)                # (B, C)
            grad_in = model.backward(grad)                 # (B, D)
            grad_pool = pooling.backward(grad_in)          # (B, T, D)
            grad_embed_with_pos = pos_embed.backward(grad_pool)
            embedding.backward(grad_embed_with_pos)        # updates d_embedding
            pos_embed.update(learning_rate)

            model.update(learning_rate)
            embedding.update(learning_rate)

        if epoch % 10 == 0:
            x_eval = embedding.forward(X_ids_orig)
            x_eval = pos_embed.forward(x_eval)
            x_eval = pooling.forward(x_eval, mask_orig)
            preds_logits = model.forward(x_eval)
            probs = softmax(preds_logits)
            preds = np.argmax(probs, axis = 1)
            acc = accuracy(preds, labels_orig)
            mean_conf = np.mean(probs[np.arange(len(labels_orig)), labels_orig])
            print(f"Epoch {epoch:3d}: Loss={loss:.4f}, Accuracy={acc:.2f}, Conf={mean_conf:.4f}")


# Visualize training with some sentences
id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech",}

test_texts = dataset["test"]["text"][:20]
test_labels = dataset["test"]["label"][:20]

# visualize_predictions(model, embedding, pos_embed, pooling, tok, test_texts, test_labels, id2label)

custom_text = "Nasa moon jupiter"
custom_text_pred(model, embedding, pos_embed, pooling, tok, custom_text, id2label)

# np.savez("model_checkpoint.npz",
#          W1 = model.W,
#          b1 = model.b,
#          W2 = model.dW,
#          b2 = model.db,
#          embed_weights = embedding.embedding,
#          pos_weights = pos_embed.embedding)

save_dict = {
    'embed_weights' : embedding.embedding,
    'pos_weights' : pos_embed.embedding
}

# Save weight and bias from each linear layer
for i, (layer, _) in enumerate(model.layers):
    save_dict[f'W{i}'] = layer.W
    save_dict[f'b{i}'] = layer.b

np.savez("model_checkpoint.npz", **save_dict)
print("Model saved successfully")

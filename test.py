import numpy as np
import re
from collections import Counter
from datasets import load_dataset
import os
import json

class Tokenizer:
    def __init__(self, vocab_size=1000, unk_token="<UNK>", pad_token="<PAD>"):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.pad_token = pad_token

    def tokenize(self, text):
        # Convert to lowercase and split on words (removes punctuation)
        return re.findall(r"[a-z0-9]+", text.lower())
    
    def fit(self, texts):
        # memory efficiency
        freq = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            freq.update(tokens)

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

    def pad_sequences(self, seqs, max_len):
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

        return np.array(padded, dtype=np.int32), np.array(mask, dtype=np.float32)
    
    def save(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # save the word -> id mapping
        with open(os.path.join(folder_path, "word2id.json"), "w") as f:
            json.dump(self.word2id, f)

        # save the config
        config = {
            "vocab_size" : self.vocab_size,
            "unk_token" : self.unk_token,
            "pad_token" : self.pad_token
        }

        with open(os.path.join(folder_path, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, folder_path):
        # load the config
        with open(os.path.join(folder_path, "config.json"), "r") as f:
            config = json.load(f)

        # create a new tokenizer with the saved config
        tokenizer = cls(
            vocab_size = config["vocab_size"],
            unk_token = config["unk_token"],
            pad_token = config["pad_token"]
        )

        # load the word -> id mapping
        with open(os.path.join(folder_path, "word2id.json"), "r") as f:
            # rebuild the rest of the tokenizer's state
            tokenizer.word2id = json.load(f)

        # create an empty list of the correct size
        id2word_list = [None] * len(tokenizer.word2id)
        # populate the list using the loaded dictionary
        for word, idx in tokenizer.word2id.items():
            id2word_list[idx] = word
        tokenizer.id2word = id2word_list

        # rebuild the rest of the tokenizer's state
        tokenizer.pad_id = tokenizer.word2id[tokenizer.pad_token]
        tokenizer.unk_id = tokenizer.word2id[tokenizer.unk_token]

        return tokenizer

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
        clip_threshold = 1.0
        self.d_embedding = clip_gradients(self.d_embedding, clip_threshold)

        self.embedding -= learning_rate * self.d_embedding

        # ensure the padding embedding remains zero after the update
        if self.pad_id is not None:
            self.embedding[self.pad_id] = 0.0

class PositionalEmbedding:
    def __init__(self, max_len, embedding_dim):
        self.embedding = np.random.randn(max_len, embedding_dim) * 0.01

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
        clip_threshold = 1.0
        self.d_embedding = clip_gradients(self.d_embedding, clip_threshold)

        self.embedding -= learning_rate * self.d_embedding

# Self Attention
# class SelfAttention:
#     def __init__(self, embed_dim):
#         self.embed_dim = embed_dim
#         self.scale = np.sqrt(self.embed_dim).astype(np.float32)

#         # weight matrices for Q, K, V (each D x D)
#         self.W_q = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim) * 0.1
#         self.W_k = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim) * 0.1
#         self.W_v = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim) * 0.1

#         # gradients
#         self.dW_q = np.zeros_like(self.W_q)
#         self.dW_k = np.zeros_like(self.W_k)
#         self.dW_v = np.zeros_like(self.W_v)

#     def forward(self, x, mask):
#         self.x = x
#         self.mask = mask
#         B, T, D = x.shape
        
#         # compute Q, K, V
#         self.Q = x @ self.W_q
#         self.K = x @ self.W_k
#         self.V = x @ self.W_v

#         # attention scores
#         scores = self.Q @ self.K.transpose(0, 2, 1) / self.scale

#         # mask padding (set attention to -inf where mask is 0)
#         mask_expanded = mask[:, np.newaxis, :]
#         scores = np.where(mask_expanded == 0, -1e9, scores)
#         scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)

#         # softmax attention weights
#         self.attn_weights = np.exp(scores - np.max(scores, axis = 2, keepdims=True))
#         self.attn_weights /= np.sum(self.attn_weights, axis = 2, keepdims=True) + 1e-8

#         # output: weighted sum of V
#         self.out = self.attn_weights @ self.V
#         return self.out
    
#     def backward(self, grad_output):
#         # B, T, D = self.x.shape
#         B, T, D = grad_output.shape

#         # dV = attn_weights^T @ d_out
#         dV = self.attn_weights.transpose(0, 2, 1) @ grad_output
        
#         # d_attn_weights @ V^T
#         d_attn = grad_output @ self.V.transpose(0, 2, 1)

#         # softmax backward
#         dscores = self.attn_weights * (d_attn - np.sum(d_attn * self.attn_weights, axis = 2, keepdims=True))

#         # divide by scale
#         dscores /= self.scale

#         # gradients for Q and K
#         dQ = dscores @ self.K
#         dK = dscores.transpose(0, 2, 1) @ self.Q

#         # chain all back to input
#         dx_q = dQ @ self.W_q.T
#         dx_k = dK @ self.W_k.T
#         dx_v = dV @ self.W_v.T

#         dx = dx_q + dx_k + dx_v # total gradient to pass back
#         if self.mask is not None:
#             dx *= self.mask[:, :, None]

#         # Compute gradients of weights
#         self.dW_q = np.zeros_like(self.W_q)
#         self.dW_k = np.zeros_like(self.W_k)
#         self.dW_v = np.zeros_like(self.W_v)

#         for b in range(B):
#             self.dW_q += self.x[b].T @ dQ[b]  # (D, T) @ (T, D) = (D, D)
#             self.dW_k += self.x[b].T @ dK[b]
#             self.dW_v += self.x[b].T @ dV[b]

#         return dx
    
#     def update(self, lr):
#         # Calculate the L2 norm of the gradients
#         # norm_dWq = np.linalg.norm(self.dW_q)
#         # norm_dWk = np.linalg.norm(self.dW_k)
#         # norm_dWv = np.linalg.norm(self.dW_v)

#         # # We can also check the norm of the weights themselves
#         # norm_Wq = np.linalg.norm(self.W_q)

#         # print(f"Gradients Norms -> dW_q: {norm_dWq:.4f}, dW_k: {norm_dWk:.4f}, dW_v: {norm_dWv:.4f} | Weight Norm W_q: {norm_Wq:.4f}")

#         self.W_q -= lr * self.dW_q
#         self.W_k -= lr * self.dW_k
#         self.W_v -= lr * self.dW_v

class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        if embed_dim % num_heads != 0:
            raise ValueError(f"Embedding dimension {embed_dim} must be divisble by number of heads {num_heads}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # still one large Wq, Wk, Wv for initial projection
        self.W_q = np.random.randn(embed_dim, embed_dim) * 0.1
        self.W_k = np.random.randn(embed_dim, embed_dim) * 0.1
        self.W_v = np.random.randn(embed_dim, embed_dim) * 0.1

        # final linear layer to combine te outputs of all heads
        self.W_o = np.random.randn(embed_dim, embed_dim) * 0.1

    def forward(self, x, mask):
        self.x = x
        self.mask = mask
        B, T, D = x.shape # batch_size, sequence_length, embedding_dim

        # 1. Initialize linear projections
        Q = self.x @ self.W_q
        K = self.x @ self.W_k
        V = self.x @ self.W_v

        # 2. Split into multiple heads
        # reshape from (B, T, D) -> (B, T, num_heads, head_dim)
        Q_split = Q.reshape(B, T, self.num_heads, self.head_dim)
        K_split = K.reshape(B, T, self.num_heads, self.head_dim)
        V_split = V.reshape(B, T, self.num_heads, self.head_dim)

        # Transpose to (B, num_heads, T, head_dim) to perform attention per head
        self.Q_heads = Q_split.transpose(0, 2, 1, 3)
        self.K_heads = K_split.transpose(0, 2, 1, 3)
        self.V_heads = V_split.transpose(0, 2, 1, 3)

        # 3. Scaled dot-product attention for all heads in parallel
        scale = np.sqrt(self.head_dim)
        scores = self.Q_heads @ self.K_heads.transpose(0, 1, 3, 2) / scale

        if mask is not None:
            # mask needs to be reshaped for broadcasting with heads
            # from (B, T) -> (B, 1, 1, T)
            mask_expanded = mask[:, np.newaxis, np.newaxis, :]
            scores = np.where(mask_expanded == 0, -1e9, scores)

        self.attn_weights = softmax(scores, axis=-1) # softmax over the last dimension

        # get the output from each head 
        out_heads = self.attn_weights @ self.V_heads # shape : (B, num_heads, T, head_dim)
        
        # 4. Concatenate heads and apply final linear layer
        # transpose back to (B, T, num_head, head_dim)
        out_concat = out_heads.transpose(0, 2, 1, 3)
        # reshape to (B, T, D) to stitch heads together
        self.concatenated_output = out_concat.reshape(B, T, D)

        # Final projection
        final_output = self.concatenated_output @ self.W_o

        return final_output
    
    def backward(self, grad_output):
        B, T, D = grad_output.shape

        # gradient through the final linear layer (W_o)
        self.dW_o = self.concatenated_output.reshape(B*T, D).T @ grad_output.reshape(B*T, D)
        grad_concat = grad_output @ self.W_o.T

        # gradient through concatenation and transpose
        grad_concat_reshaped = grad_concat.reshape(B, T, self.num_heads, self.head_dim)
        grad_out_heads = grad_concat_reshaped.transpose(0, 2, 1, 3)

        # gradient through attention mechanism (in parallel for each head)
        # grad w.r.t V_heads
        d_V_heads = self.attn_weights.transpose(0, 1, 3, 2) @ grad_out_heads

        # grad w.r.t attn_weights
        d_attn_weights = grad_out_heads @ self.V_heads.transpose(0, 1, 3, 2)

        # grad w.r.t scores (softmax backward)
        d_scores = self.attn_weights * (d_attn_weights - np.sum(d_attn_weights * self.attn_weights, axis=-1, keepdims=True))

        # grad w.r.t Q_heads and K_heads
        scale = np.sqrt(self.head_dim)
        d_scores /= scale
        d_Q_heads = d_scores @ self.K_heads
        d_K_heads = d_scores.transpose(0, 1, 3, 2) @ self.Q_heads

        # combine gradients from all heads
        # transpose gradients back from (B, nums_heads, T, head_dim) to (B, T num_heads, head_dim)
        d_Q_split = d_Q_heads.transpose(0, 2, 1, 3)
        d_K_split = d_K_heads.transpose(0, 2, 1, 3)
        d_V_split = d_V_heads.transpose(0, 2, 1, 3)

        # reshape to (B, T, D)
        dQ = d_Q_split.reshape(B, T, D)
        dK = d_K_split.reshape(B, T, D)
        dV = d_V_split.reshape(B, T, D)

        # gradient through initial linear projections (W_q, W_k, W_v)
        self.dW_q = self.x.reshape(B*T, D).T @ dQ.reshape(B*T, D)
        self.dW_k = self.x.reshape(B*T, D).T @ dK.reshape(B*T, D)
        self.dW_v = self.x.reshape(B*T, D).T @ dV.reshape(B*T, D)

        # final gradient to pass back to the residual connection
        dx = (dQ @ self.W_q.T) + (dK @ self.W_k.T) + (dV @ self.W_v.T)

        return dx

    def update(self, lr):
        # define a clipping threshold
        clip_threshold = 1.0

        # clip each gradient before the update step
        self.dW_q = clip_gradients(self.dW_q, clip_threshold)
        self.dW_k = clip_gradients(self.dW_k, clip_threshold)
        self.dW_v = clip_gradients(self.dW_v, clip_threshold)
        self.dW_o = clip_gradients(self.dW_o, clip_threshold)

        # update with clipped gradients
        self.W_q -= lr * self.dW_q
        self.W_k -= lr * self.dW_k
        self.W_v -= lr * self.dW_v
        self.W_o -= lr * self.dW_o
    
class AttentionPooling:
    def __init__(self, embedding_dim):
        self.query = np.random.randn(1, embedding_dim) * 0.01

    def forward(self, x, mask):
        self.input = x
        self.mask = mask

        # compute similarity scores between each word and the learned query
        scores = np.matmul(x, self.query.T).squeeze(axis = -1)

        # apply mask to ignore padding tokens: set scores of PAD tokens to -inf
        if mask is not None:
            scores = np.where(mask == 1, scores, -1e9)

        # softmax to turn scores into weights that sum to 1 
        self.attn_weights = softmax(scores)

        # compute the weighted sum of word vectors
        # attn_weights is (B, T), input is (B, T, D)
        # we need to reshape weights to (B, T, 1) for broadcasting
        return np.sum(self.input * self.attn_weights[..., np.newaxis], axis=1)

    def backward(self, grad_output):
        # grad_output is (B, D)
        # we need to compute grad_input (B, T, D) and grad_query (1, D)

        # gradient w.r.t the input vectors
        # each input vector's gradient is the output gradient scaled by its attention weight
        grad_input = self.attn_weights[..., np.newaxis] * grad_output[:, np.newaxis, :]

        # gradient for the query vector

        # gradient of loss w.r.t the weighted sum
        d_weighted_sum = grad_output[:, np.newaxis, :] * self.input #(B, T, D)

        # derivative of softmax
        s = self.attn_weights[..., np.newaxis] #(B, T, 1)
        d_softmax = s * (np.identity(self.input.shape[1]) - s.transpose((0, 2, 1))) # (B, T, T)

        # putting it all together
        self.grad_query = (d_weighted_sum.transpose((0, 2, 1)) @ d_softmax).sum(axis=(0, 2))

        #apply mask
        if self.mask is not None:
            grad_input *= self.mask[..., np.newaxis]
        
        return grad_input
    
    def update(self, lr):
        # # compute gradient of every vector
        # grad_q = np.zeros_like(self.query)

        # for b in range(self.input.shape[0]):
        #     for t in range(self.input.shape[1]):
        #         grad_q += self.attn_weights[b, t] * self.input[b, t]

        # self.query -= lr * grad_q
        self.query -= learning_rate * self.grad_query

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
        # return np.dot(x, self.W) + self.b
        # @ handles 2D and 3D both
        return self.input @ self.W + self.b
    
    def backward(self, grad_output):
        # for calculating the weight gradient dW, we need to handle the 3D case
        if self.input.ndim == 3:
            # input is (B*T, D_in), grad_output is (B, T, D_out)
            B, T, D_in = self.input.shape
            D_out = grad_output.shape[-1]

            # reshape input to (B*T, D_in) and grad_output to (B*T, D_out)
            # this treats the sequence of words as one big batch
            input_reshaped = self.input.reshape(B * T, D_in)
            grad_output_reshaped = grad_output.reshape(B * T, D_out)

            # calculate dW using standard 2D matrix multiplication
            self.dW = input_reshaped.T @ grad_output_reshaped
        else:
            # original logic ofr 2D inputs, which is still needed for final classifier
            self.dW = self.input.T @ grad_output

        # for bias gradient db, we sum over all dimensions except the last one
        self.db = np.sum(grad_output, axis = tuple(range(grad_output.ndim - 1)))

        # the gradient w.r.t the input is calculated with a simple matrix multiplication
        return grad_output @ self.W.T
    
    def update(self, learning_rate):
        # define a clipping threshold
        clip_threshold = 1.0

        # clip each gradient before the update step
        self.dW = clip_gradients(self.dW, clip_threshold)
        self.db = clip_gradients(self.db, clip_threshold)

        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

# Layer Norm
class LayerNorm:
    def __init__(self, features, eps = 1e-6):
        self.eps = eps

        # Gamma is a scale parameter, initialized to 1s
        self.gamma = np.ones((1, 1, features))

        # Beta is a shift parameter, initialized to 0s
        self.beta = np.zeros((1, 1, features))

        # Gradients for gamma and beta
        self.d_gamma = np.zeros_like(self.gamma)
        self.d_beta = np.zeros_like(self.beta)

    def forward(self, x):
        self.input = x
        # x shape B, T, D
        self.B, self.T, self.D = x.shape

        # Calculate mean and variance in each item in the batch and sequence
        self.mean = x.mean(axis = -1, keepdims = True) # shape (B, T, 1)
        self.var = x.var(axis = -1, keepdims = True) # shape (B, T, 1)

        # Normalise to mean 0, var 1
        self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps) # shape (B, T, D)

        # Scale and shift
        out = self.gamma * self.x_norm + self.beta
        return out
    
    def backward(self, grad_output):
        # Gradients for the learnable parameters (gamma and beta)
        self.d_beta = np.sum(grad_output, axis = (0, 1), keepdims = True)
        self.d_gamma = np.sum(grad_output * self.x_norm, axis = (0, 1), keepdims = True)
        
        # Gradient for the input x
        # chain rule back through the normalization formula
        dx_norm = grad_output * self.gamma

        dvar = np.sum(dx_norm * (self.input - self.mean) * -0.5 * (self.var + self.eps) ** (-1.5), axis = -1, keepdims=True)
        dmean = np.sum(dx_norm * -1 / np.sqrt(self.var + self.eps), axis=-1, keepdims=True) - 2 * dvar * np.mean(self.input - self.mean, axis=-1, keepdims=True)

        dx = (dx_norm / np.sqrt(self.var + self.eps)) + (dvar * 2 * (self.input - self.mean) / self.D) + (dmean / self.D)

        return dx
    
    def update(self, learning_rate):
        clip_threshold = 1.0
        self.d_gamma = clip_gradients(self.d_gamma, clip_threshold)
        self.d_beta = clip_gradients(self.d_beta, clip_threshold)

        self.gamma -= learning_rate * self.d_gamma
        self.beta -= learning_rate * self.d_beta

class FeedForward:
    """A position wise feed forward network"""
    # the first layer expands the dimensions (eg. 32 -> 128)
    def __init__(self, embed_dim, ffn_dim):
        self.layer1 = Linear(embed_dim, ffn_dim)
        self.relu = ReLU()

        # the second layer contracts it back to the original dimension (128 -> 32)
        self.layer2 = Linear(ffn_dim, embed_dim)

    def forward(self, x):
        # data flows through layer 1, ReLU, then layer 2
        return self.layer2.forward(self.relu.forward(self.layer1.forward(x)))
    
    def backward(self, grad_output):
        # gradients flow backward in the reverse order
        grad_l2 = self.layer2.backward(grad_output)
        grad_relu = self.relu.backward(grad_l2)
        grad_l1 = self.layer1.backward(grad_relu)
        return grad_l1
    
    def update(self, learning_rate):
        self.layer1.update(learning_rate)
        self.layer2.update(learning_rate)

class Dropout:
    def __init__(self, p=0.5):
        # p is the probability of "dropping" or setting a neuron's output to zero
        if not(0.0 <= p < 1.0):
            raise ValueError("Dropout probabilty must be in the range [0,1).")
        self.p = p
        self.is_training = True # controls whether dropout is active

    def forward(self, x):
        # Only apply dropout during training
        if self.is_training:
            self.mask = (np.random.rand(*x.shape) > self.p)
            # Apply the mask and scale the result
            # Scale by 1/(1-p) to keep the expected output value the same
            return x * self.mask / (1.0 - self.p) 
        else:
            # During revaluation/prediction, do nothing
            return x
    
    def backward(self, grad_output):
        # only apply dropout's affect on gradients during training
        if self.is_training:
            return grad_output * self.mask / (1.0 - self.p)
        else:
            return grad_output
        
    def update(self, learning_rate):
        # drop has no learnable weights (like W or b), so update does nothing
        pass


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


def softmax(logits, axis = -1):
    logits_stable = logits - np.max(logits, axis=axis, keepdims=True)
    exp_scores = np.exp(logits_stable)
    probs = exp_scores / np.sum(exp_scores, axis=axis, keepdims=True)
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

def clip_gradients(grad, threshold):
    """clips gradients by norm"""
    norm = np.linalg.norm(grad)
    if norm > threshold:
        grad = grad * (threshold / norm)
    return grad

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
def custom_text_pred(model, embedding, pos_embed, norm1, attention, norm2, ffn, pooling, dropout, tok, custom_text, id2label):
    # set dropout to evaluation mode
    dropout.is_training = False

    # Tokenize
    X_ids, mask = tok.encode_batch([custom_text], max_len=20)

    # Embedding + Positional Embedding
    x_embed = embedding.forward(X_ids)
    x_embed = pos_embed.forward(x_embed)

    x_norm1 = norm1.forward(x_embed)
    x_attended_sublayer = attention.forward(x_norm1, mask)
    x_attended = x_embed + x_attended_sublayer

    x_norm2 = norm2.forward(x_attended)
    x_ffn_sublayer = ffn.forward(x_norm2)
    x_ffn = x_attended + x_ffn_sublayer

    # Pooling
    x_pooled = pooling.forward(x_ffn, mask)
    x_dropped_out = dropout.forward(x_pooled)

    # Predict
    logits = model.forward(x_dropped_out)
    probs = softmax(logits)

    # Get predicted label
    pred = np.argmax(probs, axis=1)[0]
    pred_label = id2label[pred]

    # Show prediction
    print(f"Input Sentence  : {custom_text}")
    print(f"Predicted Label : {pred_label}")
    print("Confidence      :", {id2label[i]: round(float(p), 4) for i, p in enumerate(probs[0])})

# Main

# load data
dataset = load_dataset("ag_news")
train_data = dataset["train"].select(range(10000))
texts = [item["text"] for item in train_data]
labels = [item["label"] for item in train_data]

# checkpoint and tokenizer loading/fitting
checkpoint_path = "model_checkpoint.npz"
tokenizer_path = "tokenizer_files"
current_num_samples = len(texts)

if os.path.exists(checkpoint_path) and os.path.exists(tokenizer_path):
    print("Loading tokenizer from file...")
    tok = Tokenizer.load(tokenizer_path)
    should_train = True # assume we might need to retrain, check model checkpoint next

else:
    print("No tokenizer found. Fitting a new one.")
    tok = Tokenizer()
    tok.fit(texts) # memory efficient version
    tok.save(tokenizer_path)
    should_train = True # we must train if we made a new tokenizer

# Encode + pad to a fixed length
X_ids, mask = tok.encode_batch(texts, max_len = 20)
labels = np.array(labels)

# Build model pieces using existing classes
vocab_size = len(tok.id2word)
embedding_dim = 16
num_heads = 4 # 16 is divisible by 4
ffn_dim = 128 # usually 2-4 times embed_dim
hidden_dim = 64
num_classes = 4
max_len = X_ids.shape[1]

# Embedding + Pooling setup
embedding = Embedding(vocab_size, embedding_dim, pad_id=tok.pad_id)
pos_embed = PositionalEmbedding(max_len=max_len, embedding_dim=embedding_dim)
attention = MultiHeadAttention(embedding_dim, num_heads)
ffn = FeedForward(embedding_dim, ffn_dim)
norm1 = LayerNorm(embedding_dim)
norm2 = LayerNorm(embedding_dim)
# pooling = MaskedMeanPooling()
dropout = Dropout(p=0.5)
pooling = AttentionPooling(embedding_dim)

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

epochs = 100
learning_rate = 0.0002
batch_size = 2
checkpoint_path = "model_checkpoint.npz"
current_num_samples = len(X_ids)
should_train = True
best_acc = 0.0 # Initiate best accuracy

# check if model exists and matches current dataset
if os.path.exists("model_checkpoint.npz"):
    data = np.load("model_checkpoint.npz")
    # check if number of samples match
    if "num_samples" in data and data["num_samples"] == current_num_samples:
        # embeddings
        embedding.embedding = data["embed_weights"]
        pos_embed.embedding = data["pos_weights"]
        # attention
        attention.W_q = data["attn_Wq"]
        attention.W_k = data["attn_Wk"]
        attention.W_v = data["attn_Wv"]
        attention.W_o = data["attn_Wo"]
        # norm layers
        norm1.gamma = data["norm1_gamma"]
        norm1.beta = data["norm1_beta"]
        norm2.gamma = data["norm2_gamma"]
        norm2.beta = data["norm2_beta"]
        # ffn
        ffn.layer1.W = data["ffn_W1"]
        ffn.layer1.b = data["ffn_b1"]
        ffn.layer2.W = data["ffn_W2"]
        ffn.layer2.b = data["ffn_b2"]
        # pooling
        pooling.query = data["pool_query"]
        # final classifier
        for i, (layer, _) in enumerate(model.layers):
            layer.W = data[f"W{i}"]
            layer.b = data[f"b{i}"]
        print("Checkpoint loaded. Skipped training.")
        should_train = False
    else:
        print("Checkpoint invalid (dataset changed). Retraining...")
        should_train = True
else:
    print("No checkpoint found. Training model from scratch.")
    should_train = True

if should_train:
    dropout.is_training = True # ensure dropout is ON for training
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
            
            # Forward Pass
            x_embed = embedding.forward(xb)      # (B, T, D)
            x_embed = pos_embed.forward(x_embed) # (B, T, D)

            # residual connection start
            # sub layer 1: self attention
            x_norm1 = norm1.forward(x_embed)
            x_attended_sublayer = attention.forward(x_norm1, mb)
            x_attended = x_embed + x_attended_sublayer # first residual connection
            # sub layer 2: feed-forward network
            x_norm2 = norm2.forward(x_attended)
            x_ffn_sublayer = ffn.forward(x_norm2)
            x_ffn = x_attended + x_ffn_sublayer # second residual connection
            # residual connection end

            x_pooled = pooling.forward(x_ffn, mb) # (B, D)
            x_dropped_out = dropout.forward(x_pooled)
            logits = model.forward(x_dropped_out)     # (B, C)
            # logits -= np.max(logits, axis = 1, keepdims=True)
            probs = softmax(logits)              # (B, C)
            loss = cross_entropy_batch(probs, yb)

            # Backward Pass
            grad = gradient_loss(probs, yb)                # (B, C)
            grad_in = model.backward(grad)                 # (B, D)
            grad_dropout = dropout.backward(grad_in)
            grad_pool = pooling.backward(grad_dropout)          # (B, T, D)

            # backward through ffn sublayer
            grad_ffn_sublayer = grad_pool
            grad_ffn_skip = grad_pool
            grad_from_ffn = ffn.backward(grad_ffn_sublayer)
            grad_norm2 = norm2.backward(grad_from_ffn)
            grad_from_ffn_block = grad_norm2 + grad_ffn_skip

            # backward through attention sublayer
            grad_attn_sublayer = grad_from_ffn_block
            grad_attn_skip = grad_from_ffn_block
            grad_attn = attention.backward(grad_attn_sublayer)
            grad_norm1 = norm1.backward(grad_attn)
            grad_total = grad_norm1 + grad_attn_skip

            grad_embed = pos_embed.backward(grad_total)
            embedding.backward(grad_embed)        # updates d_embedding

            ffn.update(learning_rate)
            norm1.update(learning_rate)
            norm2.update(learning_rate)
            pos_embed.update(learning_rate)
            attention.update(learning_rate)
            model.update(learning_rate)
            embedding.update(learning_rate)
            pooling.update(learning_rate)

        if epoch % 10 == 0:
            dropout.is_training = False
            # start of mini-batch training
            all_preds = []
            eval_batch_size = 64 # a reasonable batch size for evaluation

            for i in range(0, len(X_ids_orig), eval_batch_size):
                # get mini batch of the original, unshuffled data
                xb_eval = X_ids_orig[i:i+eval_batch_size]
                mb_eval = mask_orig[i:i+eval_batch_size]

                # forward pass for the entire validation set
                x_eval_embed = embedding.forward(xb_eval)
                x_eval_embed = pos_embed.forward(x_eval_embed)
                
                # sublayer1: self attention
                x_eval_norm1 = norm1.forward(x_eval_embed)
                x_eval_attended_sublayer = attention.forward(x_eval_norm1, mb_eval)
                x_eval_attended = x_eval_embed + x_eval_attended_sublayer

                # sublayer2: feed-forward network
                x_eval_norm2 = norm2.forward(x_eval_attended)
                x_eval_ffn_sublayer = ffn.forward(x_eval_norm2)
                x_eval_ffn = x_eval_attended + x_eval_ffn_sublayer

                # final pooling and classification
                x_eval_pooled = pooling.forward(x_eval_ffn, mb_eval)
                x_eval_dropout = dropout.forward(x_eval_pooled)
                preds_logits = model.forward(x_eval_dropout)

                # get predictions for this batch and store them
                batch_preds = np.argmax(preds_logits, axis=1)
                all_preds.extend(batch_preds.tolist())

            # probs = softmax(preds_logits)
            # preds = np.argmax(probs, axis = 1)
            acc = accuracy(np.array(all_preds), labels_orig)
            # end of mini batch evaluation

            # mean_conf = np.mean(probs[np.arange(len(labels_orig)), labels_orig])
            print(f"Epoch {epoch:3d}: Loss={loss:.4f}, Accuracy={acc:.2f}")

            if acc > best_acc:
                best_acc = acc
                print("Saving new model...")
                save_dict = {
                    # embeddings
                    "embed_weights" : embedding.embedding,
                    "pos_weights" : pos_embed.embedding,
                    # attention
                    "attn_Wq" : attention.W_q,
                    "attn_Wk" : attention.W_k,
                    "attn_Wv" : attention.W_v,
                    "attn_Wo" : attention.W_o,
                    # norm layers
                    "norm1_gamma" : norm1.gamma,
                    "norm1_beta" : norm1.beta,
                    "norm2_gamma" : norm2.gamma,
                    "norm2_beta" : norm2.beta,
                    # ffn
                    "ffn_W1" : ffn.layer1.W,
                    "ffn_b1" : ffn.layer1.b,
                    "ffn_W2" : ffn.layer2.W,
                    "ffn_b2" : ffn.layer2.b,
                    # pooling
                    "pool_query" : pooling.query,
                    # other info
                    "num_samples" : current_num_samples,
                }

                # add final classifier weights
                for i, (layer, _) in enumerate(model.layers):
                    save_dict[f"W{i}"] = layer.W
                    save_dict[f"b{i}"] = layer.b

                np.savez("model_checkpoint.npz", **save_dict)
                tok.save(tokenizer_path)
            dropout.is_training = True # try it back ON for the next training epoch


# Visualize training with some sentences
id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech",}

test_texts = dataset["test"]["text"][:20]
test_labels = dataset["test"]["label"][:20]

# visualize_predictions(model, embedding, pos_embed, pooling, tok, test_texts, test_labels, id2label)

custom_text = "rocket jupiter nasa nasa rocket moon interest usa interest income"
dropout.is_training = False
custom_text_pred(model, embedding, pos_embed, norm1, attention, norm2, ffn, pooling, dropout, tok, custom_text, id2label)

np.savez("final_model_checkpoint.npz",
    embed_weights = embedding.embedding,
    pos_weights = pos_embed.embedding,
    num_samples = current_num_samples,
    **{f"W{i}": layer.W for i, (layer, _) in enumerate(model.layers)},
    **{f"b{i}": layer.b for i, (layer, _) in enumerate(model.layers)})
    
print("Final model saved.")
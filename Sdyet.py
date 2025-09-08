import numpy as np
import re
from collections import Counter
from datasets import load_dataset
import os
import json
import requests

class Tokenizer:
    def __init__(self, vocab_size=8000, unk_token="<UNK>", pad_token="<PAD>", start_token="<START>", end_token="<END>"):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token

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
        most_common = most_common[:self.vocab_size - 4]  # save space for UNK, PAD, START, END

        # Assign IDs
        self.id2word = [self.pad_token, self.unk_token, self.start_token, self.end_token] + [w for w, _ in most_common]
        self.word2id = {w: i for i, w in enumerate(self.id2word)}

        self.pad_id = self.word2id[self.pad_token]
        self.unk_id = self.word2id[self.unk_token]
        self.start_id = self.word2id[self.start_token]
        self.end_id = self.word2id[self.end_token]

    def encode(self, text):
        tokens = [self.word2id.get(tok, self.unk_id) for tok in self.tokenize(text)]
        return [self.start_id] + tokens + [self.end_id]

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
            "pad_token" : self.pad_token,
            "start_token" : self.start_token,
            "end_token" : self.end_token
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
            pad_token = config["pad_token"],
            start_token = config["start_token"],
            end_token = config["end_token"]
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
        tokenizer.start_id = tokenizer.word2id[tokenizer.start_token]
        tokenizer.end_id = tokenizer.word2id[tokenizer.end_token]

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
        # self.input_shape = x.shape
        self.batch_size, self.seq_len, self.embedding_dim = x.shape
        self.pos_embed = self.embedding[:self.seq_len, :]
        self.pos_embed_batched = np.tile(self.pos_embed, (self.batch_size, 1, 1))
        return x + self.pos_embed_batched

    def backward(self, grad_output):
        _, seq_len, _ = grad_output.shape
        # Gradient w.r.t the positional embedding is just the sum over batches
        self.d_embedding = np.zeros_like(self.embedding)
        for i in range(seq_len):
            self.d_embedding[i, :] = grad_output[:, i, :].sum(axis = 0)
        return grad_output
    
    def update(self, learning_rate):
        clip_threshold = 1.0
        self.d_embedding = clip_gradients(self.d_embedding, clip_threshold)

        self.embedding -= learning_rate * self.d_embedding

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

    def forward(self, x, padding_mask, look_ahead_mask = None):
        self.x = x
        self.mask = padding_mask
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

        # apply the look-ahead mask first (for decoder)
        if look_ahead_mask is not None:
            scores += look_ahead_mask #broadcasting adds it to every head and batch

        # apply the padding mask (to hide padding tokens)
        if self.mask is not None:
            # mask needs to be reshaped for broadcasting with heads
            # from (B, T) -> (B, 1, 1, T)
            # mask_expanded = self.mask[:, np.newaxis, np.newaxis, :]
            scores = np.where(self.mask == 0, -1e9, scores)

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

class DecoderBlock:
    def __init__(self, embed_dim, num_heads, ffn_dim):
        # the first sub-layer: masked multi-head attention
        self.norm1 = LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads)

        # the second sub-layer: FFN
        self.norm2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_dim)

        # dropout layers
        self.dropout1 = Dropout(p=0.1) # low dropout rate is common in transformers
        self.dropout2 = Dropout(p=0.1)

    def forward(self, x, padding_mask, look_ahead_mask):
        # sub-layer 1: masked multi-head attention
        # the input x is passed to the residual connection
        self.attn_input = x
        x_norm1 = self.norm1.forward(self.attn_input)
        # the attention layer now receives both masks
        attn_sublayer_output = self.attention.forward(x_norm1, padding_mask, look_ahead_mask)
        # dropout
        attn_sublayer_output = self.dropout1.forward(attn_sublayer_output)
        # first residual connection
        x_attended = self.attn_input + attn_sublayer_output

        # sub-layer 2: FFN
        # the output of the first sub-layer is the input to the second
        self.ffn_input = x_attended
        x_norm2 = self.norm2.forward(self.ffn_input)
        ffn_sublayer_output = self.ffn.forward(x_norm2)
        # dropout
        ffn_sublayer_output = self.dropout2.forward(ffn_sublayer_output)
        # second residual connection
        final_output = self.ffn_input + ffn_sublayer_output

        return final_output
    
    def backward(self, grad_output):
        # backward through ffn sub-layer
        grad_ffn_sublayer = grad_output
        grad_ffn_skip = grad_output
        grad_ffn_sublayer = self.dropout2.backward(grad_ffn_sublayer)
        grad_from_ffn = self.ffn.backward(grad_ffn_sublayer)
        grad_norm2 = self.norm2.backward(grad_from_ffn)
        grad_from_ffn_block = grad_norm2 + grad_ffn_skip

        # backward through attention sublayer
        grad_attn_sublayer = grad_from_ffn_block
        grad_attn_skip = grad_from_ffn_block
        grad_attn_sublayer = self.dropout1.backward(grad_attn_sublayer)
        grad_from_attn = self.attention.backward(grad_attn_sublayer)
        grad_norm1 = self.norm1.backward(grad_from_attn)
        final_grad = grad_norm1 + grad_attn_skip

        return final_grad
    
    def update(self, learning_rate):
        # update all learnable components in the block
        self.attention.update(learning_rate)
        self.ffn.update(learning_rate)
        self.norm1.update(learning_rate)
        self.norm2.update(learning_rate)
        self.dropout1.update(learning_rate)
        self.dropout2.update(learning_rate)

    def train(self):
        self.dropout1.is_training = True
        self.dropout2.is_training = True
    
    def eval(self):
        self.dropout1.is_training = False
        self.dropout2.is_training = False
    
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

def create_look_ahead_mask(seq_len):
    '''Creates a look-ahead mask for a given sequence length'''
    # create an upper triangle matrix of ones
    mask = np.triu(np.ones((seq_len, seq_len)), k = 1)
    # where mask is 1, replace with large negative number, otherwise 0
    return np.where(mask == 1, -1e9, 0.0)

def create_padding_mask(seq):
    # seq is (B, T) tensor of token IDs
    # returns a (B, 1, 1, T) mask that is 1 where seq is not 0, and 0 where it is
    return (seq != 0)[:, np.newaxis, np.newaxis, :]

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

def download_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    with open("tinyshakespeare.txt", "w", encoding="utf-8") as f:
        f.write(response.text)
    print("Download tinyshakespeare.txt")

# Main
# load data
download_shakespeare()
print("Loading custom dataset from tinyshakespeare.txt")
with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
    texts = f.read().splitlines('\n\n')

texts = texts[:2000]
# dataset = load_dataset("ag_news")
# train_data = dataset["train"].select(range(10000))
# texts = [item["text"] for item in train_data]

# checkpoint and tokenizer loading/fitting
tokenizer_path = "decoder_tokenizer_files"
checkpoint_path = "decoder_checkpoint.npz"

should_train = False

if os.path.exists(tokenizer_path):
    print("Loading tokenizer from file...")
    tok = Tokenizer.load(tokenizer_path)
else:
    print("No tokenizer found. Fitting a new one.")
    tok = Tokenizer()
    tok.fit(texts) # memory efficient version
    tok.save(tokenizer_path)
    should_train = True # we must train if we made a new tokenizer

# Encode + pad to a fixed length
# using a longer sequence length for generation
X_ids, _ = tok.encode_batch(texts, max_len = 60)

# create training inputs (x) and targets (y)
# input is everything except the last token
x_train = X_ids[:, :-1]
# target is everything except the first token
y_train = X_ids[:, 1:]

# Build model pieces using existing classes
vocab_size = len(tok.id2word)
embedding_dim = 256
num_heads = 4 # 16 is divisible by 4
ffn_dim = 1024 # usually 2-4 times embed_dim
num_layers = 2 # to stack 2 decoder blocks
max_len = 200

# Initialize all model components
embedding = Embedding(vocab_size, embedding_dim, pad_id=tok.pad_id)
pos_embed = PositionalEmbedding(max_len=max_len, embedding_dim=embedding_dim)
decoder_blocks = [DecoderBlock(embedding_dim, num_heads, ffn_dim) for _ in range(num_layers)]
final_layer = Linear(embedding_dim, vocab_size) # predicts a score for every word in the vocab

checkpoint_path = "decoder_checkpoint.npz"
should_train = True

if os.path.exists(checkpoint_path):
    print("Loading model weights from checkpoint...")
    data = np.load(checkpoint_path)
    try:
        embedding.embedding = data["embedding"]
        pos_embed.embedding = data["pos_embed"]
        for i, block in enumerate(decoder_blocks):
            block.attention.W_q = data[f'block_{i}_attn_Wq']
            block.attention.W_k = data[f'block_{i}_attn_Wk']
            block.attention.W_v = data[f'block_{i}_attn_Wv']
            block.attention.W_o = data[f'block_{i}_attn_Wo']
            block.norm1.gamma = data[f'block_{i}_norm1_gamma']
            block.norm1.beta = data[f'block_{i}_norm1_beta']
            block.norm2.gamma = data[f'block_{i}_norm2_gamma']
            block.norm2.beta = data[f'block_{i}_norm2_beta']
            block.ffn.layer1.W = data[f'block_{i}_ffn_W1']
            block.ffn.layer1.b = data[f'block_{i}_ffn_b1']
            block.ffn.layer2.W = data[f'block_{i}_ffn_W2']
            block.ffn.layer2.b = data[f'block_{i}_ffn_b2']
        final_layer.W = data['final_layer_W']
        final_layer.b = data['final_layer_b']
        should_train = False
        print("Checkpoint loaded successfully. Training skipped.")
    except KeyError:
        print("Checkpoint is invalid. Retraining...")
        should_train = True
else:
    print("No model checkpoint found. Training from scratch")

# Training Loop
if should_train:
    epochs = 100 # generation is harder, so more epochs might be needed
    learning_rate = 0.0002
    batch_size = 32 # can use a larger batch size

    # set model to training mode
    for block in decoder_blocks:
        block.train()

    # shuffle data
    indices = np.random.permutation(len(x_train))
    x_train = x_train[indices]
    y_train = y_train[indices]

    for epoch in range(epochs):
        # mini-batches
        for i in range(0, len(x_train), batch_size):
            xb = x_train[i:i+batch_size]           # (B, T)
            yb = y_train[i:i+batch_size]               # (B,)
            
            # create masks for the decoder
            padding_mask = create_padding_mask(xb)
            look_ahead_mask = create_look_ahead_mask(xb.shape[1])

            # Forward Pass
            x = embedding.forward(xb)      # (B, T, D)
            x = pos_embed.forward(x) # (B, T, D)
            for block in decoder_blocks:
                x = block.forward(x, padding_mask, look_ahead_mask)
            logits = final_layer.forward(x)

            # calculate loss
            B, T, V = logits.shape
            logits_reshaped = logits.reshape(B * T, V)
            yb_reshaped = yb.reshape(B * T)
            probs = softmax(logits_reshaped)
            loss = cross_entropy_batch(probs, yb_reshaped)
        
            # Backward Pass
            grad = gradient_loss(probs, yb_reshaped)                # (B, C)
            grad = grad.reshape(B, T, V)

            grad = final_layer.backward(grad)
            for block in reversed(decoder_blocks):
                grad = block.backward(grad)
            grad = pos_embed.backward(grad)
            embedding.backward(grad)

            # Update step
            final_layer.update(learning_rate)
            for block in decoder_blocks:
                block.update(learning_rate)
            pos_embed.update(learning_rate)
            embedding.update(learning_rate)

        print(f"Epoch {epoch:2d}: Loss={loss:4f}")
        if (epoch + 1) % 10 == 0:
            print(f"--- Saving checkpoint at epoch {epoch} ---")
            save_dict = {
                'embedding': embedding.embedding,
                'pos_embed': pos_embed.embedding,
                'final_layer_W': final_layer.W,
                'final_layer_b': final_layer.b,
            }
            for i, block in enumerate(decoder_blocks):
                save_dict[f'block_{i}_attn_Wq'] = block.attention.W_q
                save_dict[f'block_{i}_attn_Wk'] = block.attention.W_k
                save_dict[f'block_{i}_attn_Wv'] = block.attention.W_v
                save_dict[f'block_{i}_attn_Wo'] = block.attention.W_o
                save_dict[f'block_{i}_norm1_gamma'] = block.norm1.gamma
                save_dict[f'block_{i}_norm1_beta'] = block.norm1.beta
                save_dict[f'block_{i}_norm2_gamma'] = block.norm2.gamma
                save_dict[f'block_{i}_norm2_beta'] = block.norm2.beta
                save_dict[f'block_{i}_ffn_W1'] = block.ffn.layer1.W
                save_dict[f'block_{i}_ffn_b1'] = block.ffn.layer1.b
                save_dict[f'block_{i}_ffn_W2'] = block.ffn.layer2.W
                save_dict[f'block_{i}_ffn_b2'] = block.ffn.layer2.b
            np.savez(checkpoint_path, **save_dict)
            tok.save(tokenizer_path)
    
# Text Generation (Inference)
def generate(prompt, max_tokens = 30, temperature = 1.8, top_p = 0.95):
    # set model to evaluation mode
    for block in decoder_blocks:
        block.eval()

    tokens = [tok.start_id] + [tok.word2id.get(w, tok.unk_id) for w in tok.tokenize(prompt)]

    for _ in range(max_tokens):
        x_gen = np.array([tokens])

        look_ahead_mask = create_look_ahead_mask(x_gen.shape[1])
        padding_mask = create_padding_mask(x_gen)

        # Forward pass
        x = embedding.forward(x_gen)
        x = pos_embed.forward(x)
        for block in decoder_blocks:
            x = block.forward(x, padding_mask, look_ahead_mask)
        logits = final_layer.forward(x)

        last_word_logits = logits[0, -1, :]
        #higher temp makes it more random, lower temp makes it more confident
        if temperature > 0:
            last_word_logits = last_word_logits / temperature

        probs = softmax(last_word_logits)
        # top-p (nucleus)
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        # find the cutoff point
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff_index = np.where(cumulative_probs > top_p)[0][0]

        # keep only the top probabilities that meet the threshold
        top_p_indices = sorted_indices[:cutoff_index + 1]
        top_p_probs = probs[top_p_indices]

        # re-normalize to make sure they sum to 1
        top_p_probs = top_p_probs / np.sum(top_p_probs)

        next_token_id = np.random.choice(top_p_indices, p = top_p_probs)
        # next_token_id = np.argmax(probs)

        if next_token_id == tok.end_id:
            break

        tokens.append(int(next_token_id))

    generated_text = " ".join([tok.id2word[i] for i in tokens])
    return generated_text

# Visualize training with some sentences
prompt1 = "Love is a"
generated_text1 = generate(prompt1)
print("-"*50)
print(f"Prompt: '{prompt1}'")
print(f"Generated text: {generated_text1}")

prompt2 = "Hate is"
generated_text2 = generate(prompt2)
print("-"*50)
print(f"Prompt: '{prompt2}'")
print(f"Generated text: {generated_text2}")
                

           
# # Visualize training with some sentences
# id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech",}

# test_texts = dataset["test"]["text"][:20]
# test_labels = dataset["test"]["label"][:20]

# # visualize_predictions(model, embedding, pos_embed, pooling, tok, test_texts, test_labels, id2label)

# custom_text = "rocket jupiter nasa nasa rocket moon interest usa interest income"
# dropout.is_training = False
# custom_text_pred(model, embedding, pos_embed, norm1, attention, norm2, ffn, pooling, dropout, tok, custom_text, id2label)

# np.savez("final_model_checkpoint.npz",
#     embed_weights = embedding.embedding,
#     pos_weights = pos_embed.embedding,
#     num_samples = current_num_samples,
#     **{f"W{i}": layer.W for i, (layer, _) in enumerate(model.layers)},
#     **{f"b{i}": layer.b for i, (layer, _) in enumerate(model.layers)})
    
# print("Final model saved.")
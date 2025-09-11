import re
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# read the raw text
with open("the-verdict.txt", "r", encoding="utf8") as f:
    raw_text = f.read()

class GPTdataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # tokenize the entire text
        token_ids = tokenizer.encode(txt)

        # use a sliding window to chunk the book into overlapping input/target sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader(txt, batch_size = 4, max_length = 256, stride = 128, shuffle = True, drop_last = True):

    # initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # create dataset
    dataset = GPTdataset(txt, tokenizer, max_length, stride)

    # create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
    )

    return dataloader

# model configuration
# these values are for a GPT-2 small (124M) model
model_config = {
    "vocab_size": 50257,    # vocabulary size
    "context_length": 256, # context length (smaller right now fro training)
    "emb_dim": 768,         # embedding dimension
    "n_heads": 12,          # number of attention heads
    "n_layers": 12,         # number of layers
    "drop_rate": 0.1,       # dropout rate
    "qkv_bias": False,      # query-key-value bias
}

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length):
        super().__init__()
        self.d_out = d_out

        # linear layers to create the Query, Key and Value vectors
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)

        # register the casual mask as a buffer
        # 'register_buffer' makes it part of the model's state but not a trainable parameter
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal = 1))

    def forward(self, x):
        # x is our input embeddings with shape (batch_size, sequence_length, emd_dim)
        b, sequence_length, d_in = x.shape
        # create Q, K, V vectors for all tokens in the batch
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # calculate attention scores by taking the dot product of queries and keys
        # we transpose the last two dimensions of 'keys' for matrix multiplication
        attn_scores = queries @ keys.transpose(1, 2)

        # apply casual mask before softmax
        # we use the mask up to the current sequence length to be flexible
        attn_scores.masked_fill_(self.mask.bool()[:sequence_length, :sequence_length], -torch.inf)

        # apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores / self.d_out**0.5, dim = -1)

        # compute final context vectors by multiplying weights with values
        context_vectors = attn_weights @ values

        return context_vectors

# embedding layers
# token embedding
# this is essentially a lookup table. For each token ID, it provides a learned vector
token_embedding_layer = torch.nn.Embedding(model_config["vocab_size"], model_config["emb_dim"])

# positional embedding
# this is another lookup table. For each position (0, 1, 2..), it provides a learned vector
pos_embedding_layer = torch.nn.Embedding(model_config["context_length"], model_config["emb_dim"])

dataloader = create_dataloader(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

# get the token embeddings for our batch
token_embeddings = token_embedding_layer(inputs)

# get the positional embeddings for the sequence length
sequence_length = inputs.shape[1]
pos_embeddings = pos_embedding_layer(torch.arange(sequence_length))

# add them together to create the final input embeddings
# pytorch broadcasting automatically adds the pos_embeddings to each sequence in the batch
input_embeddings = token_embeddings + pos_embeddings

# create an instance of the SelfAttention layer
# the input and output dimensions will be our embedding dimensions
self_attention = SelfAttention(model_config["emb_dim"], model_config["emb_dim"], model_config["context_length"])
context_vectors = self_attention(input_embeddings)

print("Shape of input embeddings:", input_embeddings.shape)
print("Shape of output context vectors:", context_vectors.shape)


# ================================================================================================
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
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout = 0.1, qkv_bias = False):
        super().__init__()
        # ensure the output dimension is divisible by the number of heads
        assert d_out % num_heads == 0, "d_out must be divisible num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # the dimension for each head's Q, K, V

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_in, d_out) # final linear layer
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, sequence_length, d_in = x.shape

        # create Q, K, V vectors
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # reshape and transpose for multi-head processing
        # original shape:(batch, seq_len, emb_dim)
        # new shape: (batch, num_heads, seq_len, head_dim)
        queries = queries.view(b, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        # compute attention scores
        attn_scores = queries @ keys.transpose(2, 3) # transpose last two dimensions

        # apply causal mask
        attn_scores.masked_fill_(self.mask.bool()[:sequence_length, :sequence_length], -torch.inf)

        # compute attention weights and apply dropout
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim = -1)
        attn_weights = self.dropout(attn_weights)

        # compute context vectors
        context_vec = (attn_weights @ values).transpose(1, 2)

        # combine heads back into a single tensor
        context_vec = context_vec.contiguous().view(b, sequence_length, self.d_out)

        # pass through the final linear layer
        return self.out_proj(context_vec)
    
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim = -1, keepdim=True, unbiased = False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi))* (x + 0.044715 * torch.pow(x, 3))
        ))
    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), # expansion
            GELU(), # activation
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]), # contraction
        )

    def forward(self, x):
        return self.layers(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"]
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # first sub-layer: MultiHeadAttention
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_resid(x)
        x = x + shortcut # add residual connection

        # second sub-layer: Feed-Forward Network
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut # add residual connection

        return x

# the complete gpt model architecture
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # input layers
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # a sequence of transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # output layers
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        # create token and positional embeddings
        tok_embeds = self.token_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds # shape: (batch_size, seq_len, emb_dim)
        x = self.drop_emb(x)

        # pass through te transformer blocks
        x = self.trf_blocks(x)

        # get the final logits
        x = self.final_norm(x)
        logits = self.out_head(x) # shape: (batch_size, deq_len, vocab_size)

        return logits


# embedding layers
# token embedding
# this is essentially a lookup table. For each token ID, it provides a learned vector
# token_embedding_layer = torch.nn.Embedding(model_config["vocab_size"], model_config["emb_dim"])

# positional embedding
# this is another lookup table. For each position (0, 1, 2..), it provides a learned vector
# pos_embedding_layer = torch.nn.Embedding(model_config["context_length"], model_config["emb_dim"])

# create a dataloader instance
dataloader = create_dataloader(raw_text, batch_size=8, max_length=model_config["context_length"], stride=model_config["context_length"], shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

# create a model instance
model = GPTModel(model_config)

# run a batch of data through the model
logits = model(inputs)

# get the token embeddings for our batch
# token_embeddings = token_embedding_layer(inputs)

# get the positional embeddings for the sequence length
# sequence_length = inputs.shape[1]
# pos_embeddings = pos_embedding_layer(torch.arange(sequence_length))

# add them together to create the final input embeddings
# pytorch broadcasting automatically adds the pos_embeddings to each sequence in the batch
# input_embeddings = token_embeddings + pos_embeddings

# create an instance of MultiHeadAttention
# multihead_attn = MultiHeadAttention(
#     d_in = model_config["emb_dim"],
#     d_out = model_config["emb_dim"],
#     context_length = model_config["context_length"],
#     num_heads = model_config["n_heads"]
# )

# create an instance of TransformerBlock
# transformer_block = TransformerBlock(model_config)

# run the input embeddings through the block
# output = transformer_block(input_embeddings)


print("Shape of input embeddings:", inputs.shape)
print("Shape of output context vectors from transkwer:", logits.shape)


# ================================================================================================
    

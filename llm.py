import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers import GPT2TokenizerFast

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


    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout=0.1):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Enable biases to match GPT-2
        self.W_query = nn.Linear(d_in, d_out, bias=True)
        self.W_key   = nn.Linear(d_in, d_out, bias=True)
        self.W_value = nn.Linear(d_in, d_out, bias=True)
        self.out_proj = nn.Linear(d_out, d_out, bias=True)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, seq_len, _ = x.shape

        # Q, K, V
        q = self.W_query(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn_scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        attn_scores.masked_fill_(self.mask[:seq_len, :seq_len].bool(), float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Weighted sum
        context = (attn_probs @ v).transpose(1, 2).contiguous().view(b, seq_len, self.d_out)
        return self.out_proj(context)

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim, eps=1e-5)

    def forward(self, x):
        return self.norm(x)
    
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
    
def load_weights_from_gpt2(our_model, gpt2_model):
    our_state_dict = our_model.state_dict()
    gpt2_state_dict = gpt2_model.state_dict()

    # mapping embeddings and layers
    simple_map = {
        'token_emb.weight': 'transformer.wte.weight',
        'pos_emb.weight': 'transformer.wpe.weight',
        'out_head.weight': 'lm_head.weight',
        'final_norm.norm.weight': 'transformer.ln_f.weight',
        'final_norm.norm.bias': 'transformer.ln_f.bias',
    }
    for our_key, gpt2_key in simple_map.items():
        if gpt2_key in gpt2_state_dict:
            our_state_dict[our_key].copy_(gpt2_state_dict[gpt2_key])

    # map transformer blocks
    for i in range(len(our_model.trf_blocks)):
        emb_dim = our_model.token_emb.weight.shape[1]
        # QKV weights and biases
        qkv_w = gpt2_state_dict[f'transformer.h.{i}.attn.c_attn.weight']
        qkv_b = gpt2_state_dict[f'transformer.h.{i}.attn.c_attn.bias']

        # GPT2 has QKV in a combined matrix, so split it
        q_w, k_w, v_w = qkv_w.split(emb_dim, dim=1)
        q_b, k_b, v_b = qkv_b.split(emb_dim, dim=0)

        our_state_dict[f'trf_blocks.{i}.att.W_query.weight'].copy_(q_w.T)
        our_state_dict[f'trf_blocks.{i}.att.W_key.weight'].copy_(k_w.T)
        our_state_dict[f'trf_blocks.{i}.att.W_value.weight'].copy_(v_w.T)

        our_state_dict[f'trf_blocks.{i}.att.W_query.bias'].copy_(q_b)
        our_state_dict[f'trf_blocks.{i}.att.W_key.bias'].copy_(k_b)
        our_state_dict[f'trf_blocks.{i}.att.W_value.bias'].copy_(v_b)

        # Attention output projection
        our_state_dict[f'trf_blocks.{i}.att.out_proj.weight'].copy_(gpt2_state_dict[f'transformer.h.{i}.attn.c_proj.weight'].T)
        our_state_dict[f'trf_blocks.{i}.att.out_proj.bias'].copy_(gpt2_state_dict[f'transformer.h.{i}.attn.c_proj.bias'])

        # LayerNorm
        our_state_dict[f'trf_blocks.{i}.norm1.norm.weight'].copy_(gpt2_state_dict[f'transformer.h.{i}.ln_1.weight'])
        our_state_dict[f'trf_blocks.{i}.norm1.norm.bias'].copy_(gpt2_state_dict[f'transformer.h.{i}.ln_1.bias'])
        our_state_dict[f'trf_blocks.{i}.norm2.norm.weight'].copy_(gpt2_state_dict[f'transformer.h.{i}.ln_2.weight'])
        our_state_dict[f'trf_blocks.{i}.norm2.norm.bias'].copy_(gpt2_state_dict[f'transformer.h.{i}.ln_2.bias'])

        # Feed-Forward
        our_state_dict[f'trf_blocks.{i}.ff.layers.0.weight'].copy_(gpt2_state_dict[f'transformer.h.{i}.mlp.c_fc.weight'].T)
        our_state_dict[f'trf_blocks.{i}.ff.layers.0.bias'].copy_(gpt2_state_dict[f'transformer.h.{i}.mlp.c_fc.bias'])
        our_state_dict[f'trf_blocks.{i}.ff.layers.2.weight'].copy_(gpt2_state_dict[f'transformer.h.{i}.mlp.c_proj.weight'].T)
        our_state_dict[f'trf_blocks.{i}.ff.layers.2.bias'].copy_(gpt2_state_dict[f'transformer.h.{i}.mlp.c_proj.bias'])

    our_model.load_state_dict(our_state_dict)
    return our_model

# Generate Function
def generate_text(
    model,
    tokenizer,
    device,
    context,
    max_new_tokens,
    context_length,
    temperature=0.8,
    top_k=50,
    top_p=1.0,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
):
    # encode the input context
    in_ids = tokenizer.encode(context)
    in_tensor = torch.tensor(in_ids, dtype=torch.long).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        generated_tokens = in_ids.copy()  # keep track of generated token ids
        for _ in range(max_new_tokens):
            # truncate input to context length
            in_tensor_trunc = in_tensor[:, -context_length:]

            # forward pass
            logits = model(in_tensor_trunc)
            logits = logits[:, -1, :]  # last token logits

            # apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated_tokens):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            # apply no-repeat ngram filtering
            if no_repeat_ngram_size > 1 and len(generated_tokens) >= no_repeat_ngram_size - 1:
                prev_ngram = tuple(generated_tokens[-(no_repeat_ngram_size-1):])
                banned_tokens = set()
                for i in range(len(generated_tokens) - (no_repeat_ngram_size-1)):
                    if tuple(generated_tokens[i:i+(no_repeat_ngram_size-1)]) == prev_ngram:
                        banned_tokens.add(generated_tokens[i + (no_repeat_ngram_size-1)])
                for token_id in banned_tokens:
                    logits[0, token_id] = -float('inf')

            # temperature scaling
            if temperature > 0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)

                # top-k filtering
                if top_k > 0:
                    top_k = min(top_k, probs.shape[-1])
                    top_k_probs, top_k_indices = torch.topk(probs, top_k)
                    probs = torch.zeros_like(probs).scatter_(-1, top_k_indices, top_k_probs)

                # top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    sorted_probs[sorted_indices_to_remove] = 0.0
                    probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)

                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                # greedy decoding
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

            # append next token
            in_tensor = torch.cat((in_tensor, next_token_id), dim=1)
            generated_tokens.append(next_token_id.item())

    # decode the final sequence
    return tokenizer.decode(generated_tokens)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    # model configuration
    # these values are for a GPT-2 small (124M) model
    model_config = {
        "vocab_size": 50257,    # vocabulary size
        "context_length": 1024, # context length (smaller right now fro training)
        "emb_dim": 768,         # embedding dimension
        "n_heads": 12,          # number of attention heads
        "n_layers": 12,         # number of layers
        "drop_rate": 0.1,       # dropout rate
        "qkv_bias": False,      # query-key-value bias
    }

    # we'll put training-specific settings here
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_config["device"] = device
    print(f"using device: {device}")

    # initialize our model and the pre-trained GPT-2 model
    torch.manual_seed(123) # for reproducibility
    model = GPTModel(model_config)
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

    # load the weights
    print("loading weights from pre-trained GPT-2...")
    model = load_weights_from_gpt2(model, gpt2)
    model.to(device) # mode model to gpu
    print("Weights loaded successfully!")

    # testing
    # tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # start_context = "The trial had taken place in a small, remote town in the north"

    # generate 20 new tokens based on the start_context
    generated_text = generate_text(
        model, tokenizer, device,
        context="The trial had taken place in a small, remote town in the north",
        max_new_tokens=60,
        context_length=128,
        temperature=1.0,
        top_k=100,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
    )

    print("\n--- Generated Text ---")
    print(generated_text)

    # ================================================================================================
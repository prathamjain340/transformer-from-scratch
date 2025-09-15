import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import GPT2LMHeadModel

# read the raw text
with open("tinyshakespeare.txt", "r", encoding="utf8") as f:
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

def calculate_loss(model, dataloader, device):
    model.eval() # set model to evaluation mode
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            total_loss += loss.item()
    return total_loss / len(dataloader)


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
    
def load_weights_from_gpt2(our_model, gpt2_model):
    our_state_dict = our_model.state_dict()
    gpt2_state_dict = gpt2_model.state_dict()

    # mapping layer names correctly
    key_map = {
        'token_emb.weight' : 'transformer.wte.weight',
        'pos_emb.weight' : 'transformer.wpe.weight',
        'out_head.weight' : 'lm_head.weight',
        'final_norm.scale' : 'transformer.ln_f.weight',
        'final_norm.shift' : 'transformer.ln_f.bias',
    }

    # map the transformer block
    for i in range(our_model.trf_blocks.__len__()):
        key_map[f'trf_blocks.{i}.att.W_query.weight'] = f'transformer.h.{i}.attn.c_attn.weight'
        key_map[f'trf_blocks.{i}.att.W_key.weight'] = f'transformer.h.{i}.attn.c_attn.weight'
        key_map[f'trf_blocks.{i}.att.W_value.weight'] = f'transformer.h.{i}.attn.c_attn.weight'
        key_map[f'trf_blocks.{i}.att.out_proj.weight'] = f'transformer.h.{i}.attn.c_proj.weight'
        key_map[f'trf_blocks.{i}.att.out_proj.bias'] = f'transformer.h.{i}.attn.c_proj.bias'
        key_map[f'trf_blocks.{i}.norm1.scale'] = f'transformer.h.{i}.ln_1.weight'
        key_map[f'trf_blocks.{i}.norm1.shift'] = f'transformer.h.{i}.ln_1.bias'
        key_map[f'trf_blocks.{i}.ff.layers.0.weight'] = f'transformer.h.{i}.mlp.c_fc.weight'
        key_map[f'trf_blocks.{i}.ff.layers.0.bias'] = f'transformer.h.{i}.mlp.c_fc.bias'
        key_map[f'trf_blocks.{i}.ff.layers.2.weight'] = f'transformer.h.{i}.mlp.c_proj.weight'
        key_map[f'trf_blocks.{i}.ff.layers.2.bias'] = f'transformer.h.{i}.mlp.c_proj.bias'
        key_map[f'trf_blocks.{i}.norm2.scale'] = f'transformer.h.{i}.ln_2.weight'
        key_map[f'trf_blocks.{i}.norm2.shift'] = f'transformer.h.{i}.ln_2.bias'

    for our_key, gpt2_key in key_map.items():
        if "att.W_" in our_key: # handle the combined QKV weights
            # the pretrained model combines Q, K, and V weights into one tensor, we need to split it
            qkv_weights = gpt2_state_dict[gpt2_key]
            emb_dim = our_model.token_emb.weight.shape[1]
            q_w, k_w, v_w = qkv_weights.split(emb_dim, dim=1)

            if "W_query" in our_key:
                our_state_dict[our_key].copy_(q_w.mT)
            if "W_key" in our_key:
                our_state_dict[our_key].copy_(k_w.mT)
            if "W_value" in our_key:
                our_state_dict[our_key].copy_(v_w.mT)

        elif gpt2_key in gpt2_state_dict:
            # for linear layers, we need to transpose the weights
            if our_state_dict[our_key].shape == gpt2_state_dict[gpt2_key].T.shape:
                our_state_dict[our_key].copy_(gpt2_state_dict[gpt2_key].T)
            else:
                our_state_dict[our_key].copy_(gpt2_state_dict[gpt2_key])

    our_model.load_state_dict(our_state_dict)
    return our_model

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Training loop
# we'll put training-specific settings here
device = "cuda" if torch.cuda.is_available() else "cpu"
model_config["device"] = device
print(f"using device: {device}")

# create a dataloader instance
# dataloader = create_dataloader(raw_text, batch_size=8, max_length=model_config["context_length"], stride=model_config["context_length"], shuffle=False)
dataloader = create_dataloader(raw_text, batch_size=8, max_length=model_config["context_length"], stride=model_config["context_length"], shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

n = len(raw_text)
train_text = raw_text[:int(n*0.9)]
val_text = raw_text[int(n*0.9):]

# create datalaoders for train and validation sets
train_loader = create_dataloader(
    train_text,
    batch_size=8,
    max_length=model_config["context_length"],
    stride=model_config["context_length"],
    shuffle=True,
    drop_last=True
)

val_loader = create_dataloader(
    val_text,
    batch_size=8,
    max_length=model_config["context_length"],
    stride=model_config["context_length"],
    shuffle=False,
    drop_last=False
)

# initialize model, optimizer, and dataloader
# initialize our model and the pre-trained GPT-2 model
model = GPTModel(model_config)
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

# load the weights
print("loading weights from pre-trained GPT-2...")
model = load_weights_from_gpt2(model, gpt2)
print("Weights loaded successfully!")
model.to(device) # mode model to gpu

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)

# the training function
def train_model(model, dataloader, val_loader, optimizer, device, num_epochs):
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train() # set the model to training mode
        total_train_loss = 0
        for inputs, targets in dataloader:
            # move data to the selected device (GPU)
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # forward pass: get model predictions
            logits = model(inputs)

            # calculate the loss
            # we need to reshape logits and targets for the loss function
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

            # backward pass: compute gradients
            # optimizer.zero_grad() # reset gradients from previous step
            loss.backward()

            # update the weights
            optimizer.step()

            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # validation phase
        avg_val_loss = calculate_loss(model, val_loader, device)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")


        # print average loss for the epoch
        # print(f"epoch {epoch+1}/{num_epochs}, average loss: {total_loss / len(dataloader):.4f}")

    return train_losses, val_losses

# run the training
train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, device, num_epochs=20)

# Generate Function

def generate_text(model, tokenizer, device, context, max_new_tokens):
    # encode the input context
    in_ids = tokenizer.encode(context)
    in_tensor = torch.tensor(in_ids).unsqueeze(0).to(device)

    # put the model in evaluation mode
    model.eval()

    # generate tokens in a loop
    with torch.no_grad(): # disable gradient calculation for efficiency
        for i in range(max_new_tokens):
            print(f'generating token {i+1}/{max_new_tokens}...')
            # get the logits from the model
            logits = model(in_tensor)

            # focus only on the last token's logits
            logits = logits[:, -1, :]

            # find the token with the highest probability (greedy decoding)
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

            # append the new token to our input sequence
            in_tensor = torch.cat((in_tensor, next_token_id) , dim=1)

        # decode the final sequence of token ID's back to text
        return tokenizer.decode(in_tensor.squeeze(0).tolist())

# testing
tokenizer = tiktoken.get_encoding("gpt2")
start_context = "Hello, I am"

# generate 20 new tokens based on the start_context
generated_text = generate_text(
    model=model,
    tokenizer=tokenizer,
    device=device,
    context=start_context,
    max_new_tokens=20
)

print("\n--- Generated Text ---")
print(generated_text)

# ================================================================================================
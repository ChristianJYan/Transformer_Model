# Imports
import tiktoken
import torch
from torch.nn import functional as F


# ### Data loading and Encoding


with open('../StarWarsScripts/AllScripts.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#Find out how many characters and which ones
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

##enc = tiktoken.get_encoding('gpt2')
#enc.n_vocab
#test = enc.encode("hello world")

#Encoding all data using the tiktoken tokenizer
##data = torch.tensor(enc.encode(text),dtype=torch.long)



stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s:[stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

#Encoding all data using the custom tokenizer
data = torch.tensor(encode(text),dtype=torch.long)

# ### Splitting Data into Training and Validation


#Split data for training and validation
train_num = int(0.9*len(data))

train_data = data[:train_num]
val_data = data[train_num:]


block_size = 256
batch_size = 64
learning_rate = 3e-4
max_iters = 5000
eval_interval = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
nembd = 384
nhead = 6
nlayer = 6
dropout = 0.2



def GetBatch(split):
    #Pick which split we should pull data from
    data = train_data if split == 'train' else val_data
    #Start of a random index in the data
    index = torch.randint(len(data) - block_size, (batch_size,))
    #Get the x and y batches. y will be our target values so we must go +1 on start and end
    #Using stack to get them in rows should be [batch_size][block_size] matrix
    x = torch.stack([data[i:i+block_size] for i in index])
    y = torch.stack([data[i+1:i+block_size + 1] for i in index])
    x, y = x.to(device), y.to(device)
    return x, y

xb,yb = GetBatch('train')

@torch.no_grad()
def EstimateLoss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = GetBatch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(torch.nn.Module):


    def __init__(self, head_size):
        super().__init__()
        self.key = torch.nn.Linear(nembd, head_size, bias=False)
        self.query = torch.nn.Linear(nembd,head_size,bias=False)
        self.value = torch.nn.Linear(nembd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape
        
        k = self.key(x)
        q = self.query(x) 

        #find the scores of the other tokens.

        wei = q @ k.transpose(-2,-1) * C**-0.5
        #make sure we never talk to the future
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(torch.nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = torch.nn.Linear(nembd, nembd)
        self.dropout = torch.nn.Dropout(dropout)
        

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out

class FeedForward(torch.nn.Module):

    def __init__(self, nembd):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(nembd,4 * nembd),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * nembd,nembd),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(torch.nn.Module):

    def __init__(self,nembd,nhead):
        super().__init__()
        head_size = nembd//nhead
        self.sa = MultiHeadAttention(nhead,head_size)
        self.ffwd = FeedForward(nembd)
        self.ln1 = torch.nn.LayerNorm(nembd)
        self.ln2 = torch.nn.LayerNorm(nembd)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
# #### Bigram Model

class BigramLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size, nembd)
        self.pos_emb_table = torch.nn.Embedding(block_size,nembd)

        self.blocks = torch.nn.Sequential(*[Block(nembd,nhead=nhead) for _ in range(nlayer)])

        self.ln_f = torch.nn.LayerNorm(nembd)
        self.lm_head = torch.nn.Linear(nembd, vocab_size)    

    def forward(self,idx, targets=None):

        B, T = idx.shape

        token_emb = self.token_embedding_table(idx)
        pos_emb = self.pos_emb_table(torch.arange(T,device=device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)

        return logits, loss
    
    def Generate(self,idx,max_new_tokens):
        #Idx will be (B,T) goal is (B,T + 1) -> (B,T + ...) -> (B,T + max_new_tokens). continue generating max_new_tokens
        for _ in range(max_new_tokens):
            #make sure the index is always -block_size last tokens
            idx_cond = idx[:,-block_size:]
            #Get predictions
            logits, loss = self(idx_cond)
            #Look only at last time step
            logits = logits[:,-1, :] #changes into (B,C)
            #Apply a softmax to get probilities
            probs = F.softmax(logits, dim=1) # still (B,C)
            #This is going to get a single sample from our probablities for each batch (B,1)
            idx_next = torch.multinomial(probs, num_samples=1)
            #add the sample index to the current sequence
            idx = torch.cat((idx,idx_next),dim = 1) # now it is (B, T + 1)
        return idx


# #### Generating and Loss


model = BigramLanguageModel()
m = model.to(device)

#a pytorch optimizer Adam
optimizer = torch.optim.AdamW(m.parameters(),lr=learning_rate)


batch_size = 32
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = EstimateLoss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


    #get batch samples
    xb, yb = GetBatch('train')

    #find the loss
    logits, loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
context = torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(m.Generate(context, max_new_tokens=500)[0].tolist()))



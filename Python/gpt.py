#Tokenizer
import tiktoken
import torch

with open('../StarWarsScripts/AllScripts.txt', 'r', encoding='utf-8') as f:
    text = f.read()
#print("length: ", len(text)) 
#print(text[:1000])

#Find out how many characters and which ones
chars = sorted(list(set(text)))
vocab_size = len(chars)
#print(''.join(chars))
#print(vocab_size)

enc = tiktoken.get_encoding('gpt2')
#enc.n_vocab
#test = enc.encode("hello world")

#Encoding all data using the tiktoken tokenizer
data = torch.tensor(enc.encode(text),dtype=torch.long)

#print(data.shape,data.dtype)
#print(data[:1000])

#Split data for training and validation
train_num = int(0.9*len(data))

train_data = data[:train_num]
val_data = data[train_num:]



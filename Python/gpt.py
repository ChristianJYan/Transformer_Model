#Tokenizer
import tiktoken

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
enc.n_vocab
test = enc.encode("hello world")

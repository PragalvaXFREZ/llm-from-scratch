import re 
import tiktoken
import torch
text = "Hello, world this is a test, or is it ?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
# removes whitespaces, if removal is warranted depends on the application and requirements. 
result = [item for item in result if item.strip()]
#print(result)
# for entire story
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
#print(len(preprocessed))
#print(preprocessed[:30])

# Creating a list of all unique tokens and sort them alphabetically
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
#print(vocab_size)

#creating vocabulary
vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
#    print(item)
    if i >= 50:
        break

# Implementing a simple text tokenizer 
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
#print(ids)

#converting token IDs back to text

#print(tokenizer.decode(ids))


# v2 replaces unkonw words with <|unk|> tokens
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
# "Hello" here is an unkown word    
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
#print(text)

#-------------------------------------------------------------------------------#
#Using BPE 

tokenizer = tiktoken.get_encoding("gpt2")
text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces" "of someunknownPlace.")
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
#print(integers)
#decoding
strings = tokenizer.decode(integers)
#print(strings)

text2 = "PragalvaExperimentsAsUsual"
integers2 = tokenizer.encode(text2)
#print(integers2)

## sliding window - section
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
#print(len(enc_text))

enc_sample = enc_text[50:]
context_size = 4
x = enc_sample [:context_size]
y = enc_sample [1:context_size+1]
#print(f"x: {x}")
#print(f"y:      {y}")

#next-token prediction task
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
#    print(context, "---->", desired)

# using pytorch 
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)  # Tokenizes the entire text
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")          # Initializes the tokenizer
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)  # Creates dataset
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,    # Drops the last batch if shorter than batch_size to avoid loss spikes
        num_workers=num_workers,  # Number of CPU processes to use for preprocessing
    )
    return dataloader


# Test the dataloader with batch size 1 and context size 4
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(
#    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
     raw_text, batch_size=1, max_length=7, stride=2, shuffle=False) # Experiment
data_iter = iter(dataloader)   # Converts dataloader into a Python iterator
first_batch = next(data_iter)  # Fetches the next entry
#print(first_batch)

second_batch = next(data_iter)
#print(second_batch) 

input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
#print(embedding_layer.weight)
#print(embedding_layer(torch.tensor([3])))
#print(embedding_layer(input_ids))

#-------------------------------------------------

vocab_size = 50257 #assuming that this comes from our BPE
output_dim = 256 
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
max_length = 4 
dataloader = create_dataloader_v1 (
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter (dataloader)
input, targets = next(data_iter)
print("Token IDs;\n", input)
print("\nInput shape: \n", input.shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)
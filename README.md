## Building-a-GPT

### GPT:
GPT stands for Generative Pre-trained Transformer. it is a family of Large Language 
AI models.

### Transformer:
Transformers are the underlying architecture of a model that powers the model. The 
Transformer architecture has shown outstanding performance in a variety of NLP 
tasks including language translation, text generation, query answering, and more.

### Transformer work in GPT:
Transformer is basic building block that enables the model to process and generate 
text. Input embedding captures semantic relationships between words or subworlds. 
Positional encoding This helps the model understand the sequential nature of the 
data. The transformer block usually consists of two main components, multi-head 
self-attention mechanism and feedforward neural network. The multi-head selfattention mechanism can capture the functionally relevant information of the model. 
The transformer block processes the information captured from the input sequence
into a feedforward neural network that enables the model to learn complex patterns 
in the data. To stabilize and speed up the training process, each transformer block is 
usually layer Includes normalization and residual connections. Increasing the depth 
of the model by stacking more transformer blocks allows the model to learn 
incremental representations of the input data and capture increasingly complex 
patterns. once the stacked transformer blocks process the input sequence, the final 
layer of the model generates the output sequence.


### Building a GPT is given step-by-step description with codebase:

#### Step-01:
import torch
import torch.nn as nn
from torch.nn import functional as F
=>Firstly, we need import pytorch library function and neural network model (nn).
with open('input.txt', 'r', encoding='utf-8') as f:
text = f.read()
print("length of dataset in characters: ", len(text))
=>This code reads text from a file called input.txt. It is presumed that this file 
includes text data, probably from the "tinyshakespeare" dataset, which is 
commonly used to train language models. len function used for length of text.

#### Step-02:
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l]) 
print(encode("hii there")) 
print(decode(encode("hii there"))) 
=> This code processes text data to generate a character vocabulary. It generates 
mappings between characters (stoi) and numbers (itos). It also defines the encode 
and decode functions, which convert texts to lists of numbers and vice versa.
chars = sorted(list(set(text))): This line selects all unique characters from the text 
data and puts them in a list.set(text) generates a set of unique characters from the 
text.list(set(text)) returns the set to a list, ensuring that unique characters are 
preserved.sorted() organizes the list alphabetically. 
vocab_size = len(chars):This line determines the size of the vocabulary, which is 
the total number of unique characters in the text. 
stoi = { ch:i for i,ch in enumerate(chars) }: This line generates a dictionary where 
each unique character is assigned a unique integer index. enumerate(chars) iterates 
through the list of unique characters and returns both the character and its 
index.{ch:i for i,ch in enumerate(chars) } creates a dictionary, mapping each 
character to its appropriate index. 
itos = { i:ch for i,ch in enumerate(chars) }: This code generates the inverse.

#### Step-03:
import torch 
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000
torch.Size([1115394]) torch.int64
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
block_size = 8
train_data[:block_size+1]
=> n = int(0.9*len(data)): This line determines the index for splitting the data into 
training and validation sets. It transforms 90% of the length of the data tensor 
(len(data)) to an integer (int(...)) by floor division. 
Train_data = data[:n] builds the training dataset by picking the top 90% of data.It 
slices the data tensor from index 0 to n while omitting the element at index n. 
Val_data = data[n:]: This line generates the validation dataset by picking the final 
10% of the data. It slices the data tensor from index n to the end, preserving the 
element at index n. 
In conclusion, this block of code divides the encoded text input into training and 
validation sets, with the training set including.

#### Step-04:
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
context = x[:t+1]
target = y[t]
print(f"when input is {context} the target: {target}")
torch.manual_seed(1337) 
batch_size = 4 
block_size = 8 
def get_batch(split):
data = train_data if split == 'train' else val_data
ix = torch.randint(len(data) - block_size, (batch_size,))
x = torch.stack([data[i:i+block_size] for i in ix]) 
y = torch.stack([data[i+1:i+block_size+1] for i in ix])
return x, y
torch.manual_seed(1337) is the random seed for reproducibility of results when 
using PyTorch's random number generator.The code function creates a tiny set of 
data for training or validation. It uses random indices to generate block_size 
sequences for inputs (x) and targets (y).
ix = torch.randint(len(data) - block_size, (batch_size,)) produces random indices to 
choose sequences from the dataset. len(data) - block_size computes the maximum 
valid beginning index to guarantee that a sequence of length block_size may be 
retrieved without exceeding its boundaries. torch.randint(...) returns a tensor of 
random numbers in the range [0, len(data) - block_size) with the shape 
(batch_size,). This produces batch_size random beginning indices for sequences. 
x = torch.stack([data[i:i+block_size] for i in ix]): This line generates the input 
tensor x by picking block_size sequences from the dataset. It utilizes list 
comprehension to loop over the random indices ix and choose sequences of length 
block_size beginning with each index. Torch.stack() creates the input tensor x by 
stacking the sequences along a new dimension. 
y = torch.stack([data[i+1:i+block_size+1].
x, y = x.to(device), y.to(device).This line sends the input-output pairs x and y to 
the chosen device ('cuda' or 'cpu') for calculation. It guarantees that the data is 
processed on the proper hardware (GPU or CPU) for the device specified 
previously. 
Return x, y: Finally, the function produces a tuple containing both the input tensor 
x and the destination tensor y. In summary, the get_batch(split) method creates a 
batch of input-output pairs from either the training or validation datasets. It 
chooses random beginning indices for sequences, retrieves input-output pairs of 
length block_size, and transfers them to the chosen device for processing. This 
function makes it easier to load data for training and evaluating the language 
model.

#### Step-05:
xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)
print('----')
for b in range(batch_size):
for t in range(block_size):
context = xb[b, :t+1]
target = yb[b, t]
print(f"when input is {context.tolist()} the target: {target}")
print(xb)

#### Step-06:
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)
class BigramLanguageModel(nn.Module):
def __init__(self, vocab_size):
super().__init__()
self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
def forward(self, idx, targets=None):
logits = self.token_embedding_table(idx) # (B,T,C)
if targets is None:
loss = None
else:
B, T, C = logits.shape
logits = logits.view(B*T, C)
targets = targets.view(B*T)
loss = F.cross_entropy(logits, targets)
return logits, loss
def generate(self, idx, max_new_tokens):
for _ in range(max_new_tokens):
logits, _ = self(idx)
logits = logits[:, -1, :] # (B, C)
probs = F.softmax(logits, dim=-1) # (B, C)
idx_next = torch.multinomial(probs, num_samples=1) 
idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
return idx
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
generated_text_indices = m.generate(idx=torch.zeros((1, 1), dtype=torch.long), 
max_new_tokens=100)
generated_text = decode(generated_text_indices[0].tolist())
print(generated_text)

=>FeedFoward nn Module is a simple linear layer followed by a non-linearity 
and defines a feedforward neural network layer used within the Transformer block. 
Block nn Module Transformer block is communication followed by computation. 
And then use bigram language model.
Now,
Bigram Language Model:
Bigram Language Model is a statistical language model that forecasts the next 
word in a sequence depending on the previous word. It specifically employs a 
conditional probability distribution to assess the likelihood of a word given the 
preceding word. A bigram model evaluates the conditional probability of a word 
given just the immediately previous word.
Bigram language model works:
Training: During training, the model examines a vast corpus of text data and 
counts the occurrences of each word pair (bigram) in it. 
Probability Estimation: After training, the model computes the conditional 
probability of each word based on its previous word. This is commonly 
accomplished using the maximum likelihood estimate (MLE) approach, which 
divides the count of each bigram by the count of the word before it in the dataset.
Prediction: Given a series of words as input, the bigram language model predicts 
the next word by calculating the conditional probability of each word given the 
preceding word and picking the word with the greatest probability.
Generation: The model may also be used to produce new text by predicting the 
next word based on the previous one. This method may be repeated to build a 
series of words that accurately simulates the language style detected in the training 
data. 
Bigram language models are simpler and more computationally efficient than more 
complicated models such as recurrent neural networks (RNNs) and transformers. 
However, they have drawbacks, such as their failure to capture long-term 
dependencies in the text and their reliance on local context (i.e., just evaluating the 
preceding word). Despite their shortcomings, bigram language models are widely 
utilized in a variety of natural language processing applications, including text 
production, machine translation, and speech recognition.

#### Step-7:
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
batch_size = 32
for steps inrange(100):
xb, yb = get_batch('train')
logits, loss = m(xb, yb)
optimizer.zero_grad(set_to_none=True) 
loss.backward() 
optimizer.step() 
print(loss.item())
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), 
max_new_tokens=500)[0].tolist()))

=>PyTorch optimizer is an object that performs the optimization algorithm to update 
the parameters (weights and biases) of a neural network during the training process. 
The optimization process involves adjusting the parameters based on the gradients 
of the loss function with respect to those parameters, aiming to minimize the loss 
and improve the model's performance.


#### Step-8:
The mathematical trick in self-attention:
torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3)) 
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, (3, 2)).float() 
c = a @ b
print('a=')
print(a)
print('--')
print('b=')
print(b)
print('--')
print('c=')
print(c
torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3)) 
a = a / torch.sum(a, 1, keepdim=True) 
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b 
print('a=')
print(a)
print('--')
print('b=')
print(b)
print('--')
print('c=')
print(c)
torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
print(x.shape)
torch.Size([4, 8, 2])
xbow = torch.zeros((B, T, C))
for b inrange(B):
for t inrange(T):
xprev = x[b, :t+1] # (t,C)
xbow[b, t] = torch.mean(xprev, 0)
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x # (B, T, T) @ (B, T, C) ----> (B, T, C)
torch.allclose(xbow, xbow2)
=>Here, x represent self-attention mechanism B is batch size T is length sequence 
and C is input embedding and wei means attention score.
Here Using Softmax for weighted aggregation,
Softmax function:The softmax function is a mathematical function that converts a 
vector of numerical inputs into a probability distribution whose total equals one. It 
accomplishes this by exponentiating each input value and then normalizing the 
output. 
Weighted Aggregation: Once you've received the probability distribution using 
the softmax function, you may use it to calculate weighted averages or sums of 
other variables. Each value is multiplied by the relevant probability from the 
softmax distribution, and the products are added together. This yields an 
aggregated value in which the more important values (those with greater 
probability) contribute more to the final outcome.
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
torch.allclose(xbow, xbow3)
torch.manual_seed(1337)
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x) # (B, T, 16)
q = query(x) # (B, T, 16)
v = value(x) # (B, T, 16)
wei = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
out = wei @ v
print(out.shape)
torch.Size([4, 8, 16])
wei[0] 
Now, Self-attention mechanism using,
Self-Attention: head_size = n_embd // n_head: This line determines the size of 
each attention head by dividing the embedding dimension by the number of heads. 
Self.SA = MultiHeadAttention(n_head, head_size): This line initializes an instance 
of the MultiHeadAttention class (defined elsewhere) with the supplied number of 
heads (n_head) and head size. This is the self-attention technique utilized in the 
transformer block.
Feedforward Neural Network: self.ffwd = FeedForward(n_embd); This line 
initializes an instance of the FeedForward class (defined elsewhere) with the 
embedding dimension (n_embd). This feedforward neural network is used after the 
self-attention mechanism to add nonlinearity and improve the model's ability to 
catch complicated patterns in data.
Layer Normalization: self.ln1 = nn.LayerNorm(n_embd): This line defines a 
layer normalization module with n_embd as its input dimension. Layer 
normalization ensures that each layer's activations are consistent across 
features.self.ln2 = nn.LayerNorm(n_embd): This line generates another layer 
normalization module.
Forward Method: def forward(self, x): This function specifies the Block module's 
forward pass. It accepts an input tensor x of the shape (B, T, C), where B is the 
batch size, T is the sequence length, and C is the embedding dimension.x = x +
self.sa(self.ln1(x)); This line implements the self-attention mechanism, followed by 
residual connection and layer normalization. The input tensor x goes through layer 
normalization (self.ln1) before being fed into the self-attention mechanism 
(self.sa). The output of the self-attention mechanism is added to the original input 
tensor (x) via a residual link. 
x = x + self.ffwd(self.ln2(x)); Similarly, this line applies the feedforward neural 
network, followed by residual connection and layer normalization.


#### Step-9:
def __init__(self, head_size):
super().__init__()
self.key = nn.Linear(n_embd, head_size, bias=False)
self.query = nn.Linear(n_embd, head_size, bias=False)
self.value = nn.Linear(n_embd, head_size, bias=False)
self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
self.dropout = nn.Dropout(dropout)
=>Here using dropout function. This defines a dropout layer whose probability is 
given by the dropout parameter. Dropout is applied to the attention weights before 
they are used to calculate the weighted total. This helps to minimize overfitting by 
randomly removing (setting to zero) some attention weights during training.


#### Step-10:
Overall hyperparameters,
batch_size = 16
This hyperparameter defines how many samples the model processes during each 
training iteration.A batch size of 16 indicates that the model adjusts its weights 
based on the average loss computed over 16 samples at a time.
block_size = 32
This hyperparameter determines the maximum context length for predictions. In 
the context of language models, it often refers to the longest sequences (in terms of 
tokens/words) that the model can process at once.Longer sequences can be 
shortened or divided into smaller chunks for processing.
max_iters = 5000
This option determines the maximum number of training iterations (or steps) that 
can be used during the training process. The training loop will continue for 5000 
iterations, at which point training will cease or until the convergence requirements 
are reached.
eval_interval = 100
This parameter controls how frequently the model's performance is tested on the 
training and validation sets during training.Every 100 iterations of training will be 
evaluated.
learning_rate = 1e-3
This hyperparameter controls the optimizer's learning rate.The learning rate 
determines the step size used during gradient descent, which influences training 
speed and stability.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
This line determines the device (GPU or CPU) for computing based on the 
availability of CUDA.If CUDA is enabled (i.e., a GPU is present), the model will 
be trained on it. Otherwise, it will train using the CPU ('cpu').
eval_iters = 200
This option determines how many iterations are utilized to evaluate (calculate loss) 
during each evaluation phase.During assessment, the model's performance is 
measured on a portion of the data, with 200 iterations each evaluation.
n_embd = 64
This hyperparameter determines the dimensionality of the embedding vectors. 
Embedding vectors are representations of tokens/words in a continuous vector 
space, whose size is determined by n_embd.
n_head = 4
This parameter controls the number of attention heads in the multi-head attention 
mechanism.Multi-head attention enables the model to focus on many sections of 
the input sequence at the same time, improving its capacity to detect dependencies.
n_layer = 4
This hyperparameter controls the number of layers in the transformer model. 
Each layer is made up of multi-head attention and feedforward neural network 
blocks, which help to learn complicated patterns in data.
dropout = 0.0
This parameter governs the dropout probability in the model's layers. 
A dropout value of 0.0 indicates that dropout has been turned off, and no units will 
be discarded during training.




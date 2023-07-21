import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.pre_tokenizers import ByteLevel
from transformers import GPT2Tokenizer, GPT2Model

def train_tokenizer(files,vocab_size,save_name):

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(vocab_size=vocab_size, show_progress=True,special_tokens=["[UNK]"])
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.train(files, trainer=trainer)
    tokenizer.save(save_name)

    return tokenizer

def load_tokenizer(path):

    tokenizer = Tokenizer.from_file(path)
    return tokenizer

def get_tolkien_data(
    tokenizer_path="/home/tau/emenier/data/GPT/Tokenizers/BigBPETolkienizer.json"):

    with open('/home/tau/emenier/workspace/miniGPT/lotr/full_text.txt',mode='r',encoding='utf-8') as f:
        text = f.read()

    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    data = torch.tensor(tokenizer.encode(text).ids)

    #tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    #vocab_size = tokenizer.vocab_size
    #data = torch.tensor(tokenizer.encode(text))

    return text, data, vocab_size, tokenizer

def get_shakespeare_data(
    tokenizer_path="/home/tau/emenier/data/GPT/Tokenizers/BPETokenizer_shekespeare.json"):

    with open('/home/tau/emenier/workspace/miniGPT/tinyshakespeare.txt',mode='r',encoding='utf-8') as f:
        text = f.read()

    if False:
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        print('Vocab : '+''.join(chars))
        itos = { k:v for k,v in enumerate(chars)}
        stoi = { v:k for k,v in enumerate(chars)}
        encoder = lambda string : [stoi[s] for s in string]
        decoder = lambda lst : ''.join([itos[i] for i in lst])

    tokenizer = load_tokenizer(tokenizer_path)
    data = torch.tensor(tokenizer.encode(text).ids)

    return text, data, tokenizer.get_vocab_size(), tokenizer
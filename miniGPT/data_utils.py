import torch

def get_shakespeare_data():

    with open('/home/tau/emenier/workspace/Transformers/tinyshakespeare.txt',mode='r',encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print('Vocab : '+''.join(chars))
    itos = { k:v for k,v in enumerate(chars)}
    stoi = { v:k for k,v in enumerate(chars)}
    encoder = lambda string : [stoi[s] for s in string]
    decoder = lambda lst : ''.join([itos[i] for i in lst])

    data = torch.tensor(encoder(text))

    return text, data, vocab_size, encoder, decoder
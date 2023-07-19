import miniGPT
import torch

savedir = '/home/tau/emenier/data/GPT/DecoderGPT/Train1/'
batch_size = 32 # how many independent sequences will we process in parallel?
max_block_size = 512 # what is the maximum context length for predictions?
lr = 3e-4
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float32
C = 768
n_heads = 12
n_layer = 9
dropout = 0.2
print(f'C : {C:}, n_head : {n_heads:}, n_layer : {n_layer:}')
text, data, vocab_size, encoder, decoder = miniGPT.data_utils.get_shakespeare_data()
data = data.to(device,torch.long)

train_dataset = miniGPT.train_utils.TextDataset(
                data[:int(0.9*len(data))],max_block_size)
val_dataset = miniGPT.train_utils.TextDataset(
                data[int(0.9*len(data)):],max_block_size)


gpt_model = miniGPT.gpt.DecoderGPT(vocab_size, C, n_layer, n_heads, 
                    max_block_size, dropout_freq=dropout).to(device,dtype)

trainer = miniGPT.train_utils.DecoderGPTtrainer(gpt_model,lr,
                    checkpoint_path=savedir,wd=1e-5)

trainer.train(train_dataset,val_dataset,batch_size=batch_size,
                epoch_length=100,val_length=10,
                patience=50,save_every=2)
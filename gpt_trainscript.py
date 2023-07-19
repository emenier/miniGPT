import miniGPT
import torch

savedir = '/home/tau/emenier/data/GPT/DecoderGPT/Train1/'
batch_size = 64 # how many independent sequences will we process in parallel?
max_block_size = 256 # what is the maximum context length for predictions?
lr = 3e-4
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float32
C = 384
n_heads = 6
n_layer = 6
dropout = 0.2

text, data, vocab_size, encoder, decoder = mygpt.data_utils.get_shakespeare_data()
data = data.to(device,torch.long)

train_dataset = mygpt.train_utils.TextDataset(
                data[:int(0.9*len(data))],max_block_size)
val_dataset = mygpt.train_utils.TextDataset(
                data[int(0.9*len(data)):],max_block_size)


gpt_model = mygpt.gpt.DecoderGPT(vocab_size, C, n_layer, n_heads, 
                    max_block_size, dropout_freq=0.2).to(device,dtype)

trainer = mygpt.train_utils.DecoderGPTtrainer(gpt_model,lr,
                    checkpoint_path=savedir,wd=1e-5)

trainer.train(train_dataset,val_dataset,batch_size=batch_size,
                epoch_length=100,val_length=10,
                patience=50,save_every=2)
import miniGPT
import torch

savedir = '/home/tau/emenier/data/GPT/DecoderGPT/SplitGPUTolkien/'
batch_size = 12 # how many independent sequences will we process in parallel?
lr = 3e-4
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float32
max_block_size = 1024 # what is the maximum context length for predictions?
n_heads = 12
C = n_heads*64
n_layer = 12
dropout = 0.1
print(f'C : {C:}, n_head : {n_heads:}, n_layer : {n_layer:}')
text, data, vocab_size, tokenizer = miniGPT.data_utils.get_tolkien_data()
data = data.to(device,torch.long)

train_dataset = miniGPT.train_utils.TextDataset(
                data[:int(0.9*len(data))],max_block_size)
val_dataset = miniGPT.train_utils.TextDataset(
                data[int(0.9*len(data)):],max_block_size)


gpt_model = miniGPT.gpt.DecoderGPT(vocab_size, C, n_layer, n_heads, 
                    max_block_size, dropout_freq=dropout,#).to(device,dtype)
                    gpus_to_split=[i for i in range(torch.cuda.device_count())]).to(device,dtype)

trainer = miniGPT.train_utils.DecoderGPTtrainer(gpt_model,lr,
                    checkpoint_path=savedir,wd=0.,parallel=False)

trainer.train(train_dataset,val_dataset,batch_size=batch_size,
                epoch_length=100,val_length=10,
                patience=5000,save_every=2)
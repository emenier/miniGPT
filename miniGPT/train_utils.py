from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from . import gpt


class TextDataset(Dataset):

    def __init__(self,data,block_size):

        self.data = data

        self.block_size = block_size

    def __len__(self):

        return len(self.data)-self.block_size

    def __getitem__(self,idx):

        dat = self.data[idx:idx+self.block_size+1]

        return dat[:-1], dat[1:]

class MyLoader():

    def __init__(self, dataset, batch_size, length):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.length = length
        self.counter = 1

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        if self.counter > len(self):
            self.counter=1
            raise StopIteration
        else:
            self.counter+=1
        idx = np.random.choice(len(self.dataset),size=self.batch_size)
        ret = self.dataset[idx[0]]
        to_stack = [[r] for r in ret]
        for ix in idx[1:]:
            for j,v in enumerate(self.dataset[ix]):
                to_stack[j].append(v)

        return (torch.stack(lst) for lst in to_stack)


def get_loaders(train, val, batch_size,epoch_length,val_length):

    return MyLoader(train,batch_size,epoch_length), \
            MyLoader(val,batch_size,val_length)


def monitor_memory():

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print('GPU : {:}, A : {:.3f}GB, R : {:.3f}GB, M : {:.3f}'.format(
                i,torch.cuda.memory_allocated(i)/1024**3,torch.cuda.memory_reserved(i)/1024**3,
                torch.cuda.max_memory_reserved(i)/1024**3
            ))



class DecoderGPTtrainer():

    def __init__(self,gpt_model,lr,checkpoint_path='',wd=0.,parallel=False):

        self.gpt_model = gpt_model
        self.opt = torch.optim.Adam(gpt_model.parameters(),lr=lr,weight_decay=wd)
        self.checkpoint_path = checkpoint_path

        self.losses, self.losses_std = [], []
        self.val_losses, self.val_losses_std = [], []
        self.parallel = parallel
        if parallel:
            self.gpt_model = nn.DataParallel(self.gpt_model)

    def train(self,train_dataset,val_dataset,batch_size=256,
                epoch_length=None,val_length=None,
                patience=50,save_every=1):
        if epoch_length is None: epoch_length = 1+len(train_dataset)//batch_size
        if val_length is None: val_length = 1+len(val_dataset)//batch_size
        train_loader, val_loader = get_loaders(
            train_dataset,val_dataset,batch_size,epoch_length,val_length)
        val_step = len(self.val_losses)
        best_val_step = val_step
        best_val_loss = float(np.inf)
        i_epoch = len(self.losses)
        while (i_epoch-best_val_step) < patience:
            
            i_epoch+=1
            # Train
            epoch_train_losses = self.evaluate_dataset(train_loader,
                desc=f'{i_epoch:} : Train ',length=epoch_length)
            self.losses.append(np.mean(epoch_train_losses))
            self.losses_std.append(np.std(epoch_train_losses))

            # Valid
            with torch.no_grad():
                epoch_val_losses = self.evaluate_dataset(val_loader, train=False,
                                    desc=f'{i_epoch:} : Val   ',length=val_length)
            mean_val_loss = np.mean(epoch_val_losses)
            self.val_losses.append(mean_val_loss)
            self.val_losses_std.append(np.std(epoch_val_losses))

            if mean_val_loss < best_val_loss:
                best_val_step = i_epoch
                best_val_loss = mean_val_loss

                self.save(name='best_model.trch')

            if i_epoch % save_every == 0:
                self.save(name='last_model.trch')

    def evaluate_dataset(self,loader,train=True,desc='',length=None):

        was_training = self.gpt_model.training
        if train: self.gpt_model.train()
        else: self.gpt_model.eval()

        count = 0

        epoch_losses = []

        pbar = tqdm(loader)
        for (x,y) in pbar:
            count += 1
            B,T = x.shape
            self.opt.zero_grad()
            logits = self.gpt_model(x)
            loss = F.cross_entropy(logits.view(B*T,-1),y.view(B*T))
            epoch_losses.append(loss.item())
            if train:
                loss.backward()
                self.opt.step()

            pbar.set_description(desc+' Loss {:.3e}'.format(
                                    np.array(epoch_losses).mean()))
        monitor_memory()
        if was_training: self.gpt_model.train()
        else: self.gpt_model.eval()
        return epoch_losses

    def get_state_dict(self):
        dic = {
            'state':self.gpt_model.module.state_dict() if self.parallel else self.gpt_model.state_dict(),
            'opt':self.opt.state_dict(),
            'losses':self.losses,'losses_std':self.losses_std,
            'val_losses':self.val_losses,'val_losses_std':self.val_losses_std
            }
        return dic

    def save(self,name):

        dic = self.get_state_dict()

        torch.save(dic,self.checkpoint_path+name)

    def load(self,name):

        if torch.cuda.is_available():
            dic = torch.load(self.checkpoint_path+name,map_location=torch.device('cuda'))
        else:
            dic = torch.load(self.checkpoint_path+name,map_location=torch.device('cpu'))

        if self.parallel:
            self.gpt_model.module.load_state_dict(dic['state'])
        else: self.gpt_model.load_state_dict(dic['state'])

        self.opt.load_state_dict(dic['opt'])
        self.losses, self.losses_std = dic['losses'], dic['losses_std']
        self.val_losses, self.val_losses_std = dic['val_losses'], dic['val_losses_std']
        return dic
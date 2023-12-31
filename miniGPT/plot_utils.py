import matplotlib.pyplot as plt
import numpy as np

def plot_losses(checkpoint):
    
    fs=16
    losses, losses_std = np.array(checkpoint['losses']), np.array(checkpoint['losses_std'])
    val_losses, val_losses_std = np.array(checkpoint['val_losses']), np.array(checkpoint['val_losses_std'])


    plt.figure(figsize=(15,5))
    ax = plt.gca()
    plt.title('Loaded model training',fontsize=fs)
    plt.semilogy(losses,label='Train')
    plt.fill_between(np.arange(len(losses)),losses-losses_std,losses+losses_std,
    color='tab:blue',alpha=0.5)
    val_x = np.linspace(0,len(losses)-1,len(val_losses))

    plt.plot(val_x,val_losses,label='Validation')
    plt.fill_between(val_x,val_losses-val_losses_std,
                    val_losses+val_losses_std,color='tab:orange',alpha=0.5)
    plt.ylabel('Loss',rotation=90,fontsize=fs)
    plt.xlabel('Training Iterations',fontsize=fs)
    ax.set_ylim(0.9*min(min(val_losses),min(losses)),None)
    plt.legend(fontsize=fs)
    
def model_numbers(gpt_model,max_block_size,n_layers,C, n_heads):

    to_test = {1e9:'B',1e6:'M',1e3:'K',1:''}

    n_params = sum([p.numel() for p in gpt_model.parameters() if p.requires_grad])

    for k,v in to_test.items():
        if n_params > k:
            rounded = int(n_params/k)
            n_params_str = f'{rounded:} '+v
            break


    print('Model Numbers : ')
    print(f'   Context   : {max_block_size:4d} tokens')
    print(f'   Embedding : {C:4d} tokens')
    print(f'   Layers    : {n_layers:4d}')
    print(f'   Heads     : {n_heads:4d}')
    print(f'   Params    : '+n_params_str)
    
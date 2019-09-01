import torch
import config as CONFIG
CUDA_AVAIL = torch.cuda.is_available()

import sys, argparse, os
import numpy as np
from tqdm import tqdm

# Parse arguments
parser=argparse.ArgumentParser(description='Training a language model for PennTreeBank Dataset')
parser.add_argument('--seed',default=12,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--data_dir',default='data/',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--model_size',default='medium',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--batch_size',default=20,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--seq_len',default=35,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--pretrained',action="store_false",required=False,help='(default=%(default)d')
args=parser.parse_args()

if args.model_size == 'medium': model_config = CONFIG.MediumConfig
elif args.model_size == 'large': model_config = CONFIG.LargeConfig
elif args.model_size == 'small': model_config = CONFIG.SmallConfig
else: model_config = CONFIG.TestConfig

print('*'*100,'\n',args,'\n','*'*100)
print('CUDA: ', CUDA_AVAIL)

# Set random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if CUDA_AVAIL: torch.cuda.manual_seed(args.seed)

from models.seq2seq import Seq2Seq, RnnLMEncoder, RnnLMDecoder
from models.attention import Attention
from dataloaders.ptb_loader import *

    
        
def create_ptb_loader(data_dir, batch_size, seq_len):
    train_data, valid_data, test_data, vocabulary_size = ptb_raw_data(data_dir)
    train_x, train_y = ptb_batcher(train_data, seq_len, mask=True)
    train_loader = ptb_loader(train_x , train_y, batch_size)
    test_x, test_y = ptb_batcher(test_data, seq_len, mask=True)
    test_loader = ptb_loader(test_x , test_y, batch_size)
    valid_x, valid_y = ptb_batcher(valid_data, seq_len, mask=True)
    valid_loader = ptb_loader(valid_x , valid_y, batch_size)
    return train_loader, test_loader, valid_loader, vocabulary_size

def save_checkpoint(state, folder='unnamed', is_best=False, filename='checkpoint.pth.tar'):
    mkdirs(os.path.join('experiments', folder))
    if is_best:
        save_path = os.path.join('experiments', 'maskmle', folder, 'model_best.pth.tar')
        print('saving best performing checkpoint..')
        torch.save(state, save_path)
    else:
        save_path = os.path.join('experiments','maskmle', folder, filename)
        torch.save(state, save_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
            
def main():
    
    mkdirs(os.path.join('experiments','maskmle', args.model_size))
    train_loader, test_loader, valid_loader, vocabulary_size = create_ptb_loader(args.data_dir, args.batch_size, args.seq_len)
    # Instantiate and init the model, and move it to the GPU
    model= Seq2Seq(vocabulary_size, model_config)
    if args.pretrained:
        model.load_pretrained_weights(os.path.join('experiments','lm', args.model_size, 'model_best.pth.tar'))
    else:
        print('NO PRETRAINED LANGUAGE MODEL!!')
        return
    if CUDA_AVAIL:
        model = model.cuda()

    criterion=torch.nn.NLLLoss()

    # Define optimizer
    optimizer=torch.optim.SGD(model.parameters(),lr= model_config.learning_rate)
    
    lr = model_config.learning_rate
    best_val_loss=np.inf
    for e in tqdm(range(model_config.max_max_epoch),desc='Epoch'):

        model=train(train_loader,model,criterion,optimizer)
        val_loss=eval(valid_loader,model,criterion)
        
        state = {
                'arch': "RnnLM",
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }
        if val_loss<best_val_loss:
            best_val_loss=val_loss
            save_checkpoint(state, folder=args.model_size, is_best=True)
        else:
            lr/= model_config.lr_decay
            optimizer=torch.optim.SGD(model.parameters(),lr=lr)
            save_checkpoint(state, folder=args.model_size)

        # Test
        test_loss= eval(test_loader,model,criterion)

        # Report
        msg='Epoch %d: \tValid loss=%.4f \tTest loss=%.4f \tTest perplexity=%.1f'%(e+1,val_loss,test_loss,np.exp(test_loss))
        tqdm.write(msg)



def train(dataloader,model,criterion,optimizer):

    # Set model to training mode (we're using dropout)
    model.train()
    # Get initial hidden and memory states
    states=model.get_initial_states()
    if CUDA_AVAIL: states = states.cuda()
    # Loop sequence length (train)
    for i, batch in enumerate(dataloader):
        x, y = batch
        if CUDA_AVAIL:
            x = x.cuda()
            y = y.cuda()
        # Truncated backpropagation
        states=model.detach(states)     # Otherwise the model would try to backprop all the way to the start of the data set
        # Forward pass
        logits,states=model.forward(x.long(),states)
        loss=criterion(logits.view(-1,model_config.vocab_size).float(),y.long().view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), model_config.max_grad_norm)
        optimizer.step()
        
    return model


def eval(dataloader,model,criterion):

    # Set model to evaluation mode (we're using dropout)
    model.eval()
    # Get initial hidden and memory states
    states=model.get_initial_states()
    if CUDA_AVAIL: states = states.cuda()
    # Loop sequence length (validation)
    total_loss=0
    num_loss=0
    for i, batch in enumerate(dataloader):
        x, y = batch
        if CUDA_AVAIL:
            x = x.cuda()
            y = y.cuda()
            
            
        # Truncated backpropagation
        states=model.detach(states)     # Otherwise the model would try to backprop all the way to the start of the data set

        # Forward pass
        logits,states=model.forward(x.long(),states)
        loss=criterion(logits.view(-1,model_config.vocab_size).float(),y.long().view(-1))

        # Log stuff
        total_loss+=loss.data.cpu().numpy()
        num_loss+=np.prod(y.size())
        
    return float(total_loss)/float(num_loss)

if __name__ == '__main__':
    # a = torch.rand((20,35)).long()
    # h = (torch.zeros(2,20,650),
    #         torch.zeros(2,20,650))
    # b = Seq2Seq(10000, model_config)
    # y =  b(a, h)
    # print(y.shape)
    # mask = generate_binary_mask(a)
    main()
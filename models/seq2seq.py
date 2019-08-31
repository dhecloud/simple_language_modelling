import torch
import torch.nn as nn
import random
from models.attention import Attention

class Seq2Seq(torch.nn.Module):

    def __init__(self,vocab_size, config):
        super(Seq2Seq, self).__init__()

        self.config = config
        self.encoder = RnnLMEncoder(vocab_size,config)
        self.decoder = RnnLMDecoder(vocab_size,config)
        self.linear= nn.Linear(config.hidden_size, vocab_size)
        
    def forward(self,x,h, teacher_forcing_ratio=0.5):
        
        max_len = x.shape[-1]
        predictions = torch.zeros(max_len, self.config.batch_size, self.config.vocab_size)
        y, (h,c) = self.encoder(x,h) # z = context vector
        # encoder_output: (batch_size, seq_len, hidden_size)
        # y: (medium-700, vocab_size)
        # h: (num_layers, batch_size, hidden_size)
        # c: (num_layers, batch_size, hidden_size)
        input = x[:,1].unsqueeze(-1)
        for t in range(1, max_len):
            output, attention_weights = self.decoder(input.long(),(h,c), y)
            prediction = self.linear(output.squeeze())
            predictions[t] = prediction
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.max(1)[1]
            input = (x[:,t].unsqueeze(-1) if teacher_force else top1.unsqueeze(-1))
        
        return y
        
    
    def load_pretrained_weights(self, path):
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['state_dict'])
        self.decoder.load_state_dict(checkpoint['state_dict'])

class RnnLMEncoder(torch.nn.Module):

    def __init__(self,vocab_size, config):
        super(RnnLMEncoder,self).__init__()

        # Configuration of our model
        self.num_layers = config.num_layers
        self.embedding_size = config.hidden_size
        self.hidden_size = config.hidden_size
        self.dropout_prob=0.5
        
        #model
        self.embed = torch.nn.Embedding(config.vocab_size, self.embedding_size)
        self.lstm = torch.nn.LSTM(self.embedding_size,self.hidden_size,self.num_layers,dropout=self.dropout_prob, batch_first=True)
        self.dropout = torch.nn.Dropout(self.dropout_prob)

        # Init weights
        init_range = 0.1
        self.embed.weight.data.uniform_(-init_range,init_range)

    def forward(self,x,h, enc_output=None, attention=None):
        
        y = self.embed(x)
        y = self.dropout(y)
        #y: (batch_size, seq_len, embedding_size)
        lstm_output , h = self.lstm(y,h)
        #lstm_output: (batch_size, seq_len, hidden_size)
        #h: (num_layers, batch_size, hidden_size), (num_layers, batch_size, hidden_size)

        return lstm_output, h

    def get_initial_states(self,batch_size):
        # Set initial hidden and memory states to 0
        return (torch.zeros(self.num_layers,batch_size,self.hidden_size),
                torch.zeros(self.num_layers,batch_size,self.hidden_size))

    def detach(self,h):
        # Detach returns a new variable, decoupled from the current computation graph
        return h[0].detach(),h[1].detach()

class RnnLMDecoder(torch.nn.Module):

    def __init__(self,vocab_size, config):
        super(RnnLMDecoder,self).__init__()

        # Configuration of our model
        self.num_layers = config.num_layers
        self.embedding_size = config.hidden_size
        self.hidden_size = config.hidden_size
        self.dropout_prob=0.5
        
        #model
        self.embed = torch.nn.Embedding(config.vocab_size, self.embedding_size)
        self.lstm = torch.nn.LSTM(self.embedding_size,self.hidden_size,self.num_layers,dropout=self.dropout_prob, batch_first=True)
        self.dropout = torch.nn.Dropout(self.dropout_prob)

        # Init weights
        init_range = 0.1
        self.embed.weight.data.uniform_(-init_range,init_range)
        self.attention = Attention(config)
        
    def forward(self,x,h, encoder_outputs):
        
        y = self.embed(x)
        y = self.dropout(y)
        #y: (batch_size, seq_len, embedding_size)
        lstm_output , h = self.lstm(y,h)
        #lstm_output: (batch_size, 1, hidden_size)
        #h: (num_layers, batch_size, hidden_size), (num_layers, batch_size, hidden_size)
        
        output, attention_weights = self.attention(lstm_output, encoder_outputs)

        return output, attention_weights

    def get_initial_states(self,batch_size):
        # Set initial hidden and memory states to 0
        return (torch.zeros(self.num_layers,batch_size,self.hidden_size),
                torch.zeros(self.num_layers,batch_size,self.hidden_size))

    def detach(self,h):
        # Detach returns a new variable, decoupled from the current computation graph
        return h[0].detach(),h[1].detach()


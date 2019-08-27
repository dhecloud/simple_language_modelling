import torch

class RnnLM(torch.nn.Module):

    def __init__(self,vocab_size, config):
        super(RnnLM,self).__init__()

        # Configuration of our model
        self.num_layers = config.num_layers
        self.embedding_size = config.hidden_size
        self.hidden_size = config.hidden_size
        self.dropout_prob=0.5
        
        #model
        self.embed = torch.nn.Embedding(config.vocab_size, self.embedding_size)
        self.lstm = torch.nn.LSTM(self.embedding_size,self.hidden_size,self.num_layers,dropout=self.dropout_prob, batch_first=True)
        self.dropout = torch.nn.Dropout(self.dropout_prob)
        self.fc = torch.nn.Linear(self.hidden_size,config.vocab_size)

        # Init weights
        init_range = 0.1
        self.embed.weight.data.uniform_(-init_range,init_range)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-init_range,init_range)

    def forward(self,x,h):
        
        y = self.embed(x)
        y = self.dropout(y)
        y , h = self.lstm(y,h)
        y = self.dropout(y)
        y = y.contiguous().view(-1,self.hidden_size)
        y = self.fc(y)
        
        return y,h

    def get_initial_states(self,batch_size):
        # Set initial hidden and memory states to 0
        return (torch.zeros(self.num_layers,batch_size,self.hidden_size),
                torch.zeros(self.num_layers,batch_size,self.hidden_size))

    def detach(self,h):
        # Detach returns a new variable, decoupled from the current computation graph
        return h[0].detach(),h[1].detach()

########################################################################################################################
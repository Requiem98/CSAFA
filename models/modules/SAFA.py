from libraries import *
import utilities as ut
    
from models.modules.GeM import ChannelGeneralizedMeanPooling

class SpatialAware(nn.Module):
    def __init__(self, in_shape, dimension = 8):
        super().__init__() 
        
        hidden = in_shape//2
        
        self.weight1 = nn.init.trunc_normal_(Parameter(torch.zeros(( in_shape, hidden, dimension ))), mean=0.0, std=0.005)
        self.bias1 = nn.init.constant_(Parameter(torch.zeros((   1, hidden, dimension   ))), 0.1)
        
        self.weight2 = nn.init.trunc_normal_(Parameter(torch.zeros((  hidden, in_shape, dimension   ))), mean=0.0, std=0.005)
        self.bias2 = nn.init.constant_(Parameter(torch.zeros((   1, in_shape, dimension ))), 0.1)
        
    def forward(self, x):
        
        w = torch.mean(x, axis=1).reshape(x.shape[0], -1) #(B, H1 X H2)
        
        w = torch.einsum('bi, ijd -> bjd', w, self.weight1) + self.bias1
        w = torch.einsum('bjd, jid -> bid', w, self.weight2) + self.bias2
        
        x = x.reshape(x.shape[0], x.shape[1], -1) #(B, CHANNELS, HIDDEN1, HIDDEN2) -> (B, CHANNELS, HIDDEN)
        
        x = torch.einsum('bci, bid -> bcd', x, w)  #(B ,CHANNELS, DIMENSION)
        
        x = x.reshape(x.shape[0], -1)
        
        return x
    

class GemSpatialAware(nn.Module):
    def __init__(self, in_shape, dimension = 8):
        super().__init__() 
        
        hidden = in_shape//2
        
        self.weight1 = nn.init.trunc_normal_(Parameter(torch.zeros(( in_shape, hidden, dimension ))), mean=0.0, std=0.005)
        self.bias1 = nn.init.constant_(Parameter(torch.zeros((   1, hidden, dimension   ))), 0.1)
        
        self.weight2 = nn.init.trunc_normal_(Parameter(torch.zeros((  hidden, in_shape, dimension   ))), mean=0.0, std=0.005)
        self.bias2 = nn.init.constant_(Parameter(torch.zeros((   1, in_shape, dimension ))), 0.1)
        
        self.cgem = ChannelGeneralizedMeanPooling()
        
    def forward(self, x):
        
        w = self.cgem(x) #(B, H1 X H2)
        
        w = torch.einsum('bi, ijd -> bjd', w, self.weight1) + self.bias1
        w = torch.einsum('bjd, jid -> bid', w, self.weight2) + self.bias2
        
        x = x.reshape(x.shape[0], x.shape[1], -1) #(B, CHANNELS, HIDDEN1, HIDDEN2) -> (B, CHANNELS, HIDDEN)
        
        x = torch.einsum('bci, bid -> bcd', x, w)  #(B ,CHANNELS, DIMENSION)
        
        x = x.reshape(x.shape[0], -1)
        
        return x
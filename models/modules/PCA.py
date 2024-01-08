from libraries import *
import utilities as ut


class LearnablePCA(nn.Module):
    def __init__(self, in_features, out_features, norm_first:bool):
        super().__init__()
  
        self.norm_first = norm_first
        self.linear = nn.Linear(in_features, out_features)

    @staticmethod
    def Center(x):
        #Convert to torch Tensor and keep the number of rows and columns
        
        if(x.dim() !=3 and x.dim() !=2):
            raise IndexError("PCA support just tensor of size: [B, N, M] and [B, N]")

        if(x.dim() == 3):
            
            mean = torch.mean(x, dim = 2).unsqueeze(-1).broadcast_to((x.shape))
            sd = torch.std(x, dim=2).unsqueeze(-1).broadcast_to((x.shape))
        
            x = (x - mean)/(sd+torch.tensor(1e-8))
            
        elif(x.dim() == 2):
            
            mean = torch.mean(x, dim = 0)
            sd = torch.std(x, dim=0)
            
            x = (x - mean)/(sd+1e-8)
        
        return x



    def forward(self, x):
        
        if(self.norm_first):
            if(x.dim() == 3):
                x = f.normalize(x, p=2, dim=2)
            elif(x.dim() == 2):
                x = f.normalize(x, p=2, dim=1)
            
        x = self.Center(x)

        x = self.linear(x)
        
        return x
    

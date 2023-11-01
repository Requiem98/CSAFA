from libraries import *
import utilities as ut


class LearnablePCA(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.eps = 1e-12
        self.linear = nn.LazyLinear(k)


    @staticmethod
    def Center(x):
        #Convert to torch Tensor and keep the number of rows and columns
        
        if(x.dim() !=3 and x.dim() !=2):
            raise IndexError("PCA support just tensor of size: [B, N, M] and [B, N]")

        if(x.dim() == 3):
            
            mean = torch.mean(x, dim = 1).reshape(-1, 1, x.shape[2])
            sd = torch.std(x, dim=1).reshape(-1, 1, x.shape[2])
        
            x = (x - mean)/(sd+1e-8)
            
        elif(x.dim() == 2):
            
            mean = torch.mean(x, dim = 0)
            sd = torch.std(x, dim=0)
            
            x = (x - mean)/(sd+1e-12)
        
        return x



    def forward(self, x):
 
        x = self.Center(x)

        x = self.linear(x)
        
        return x


class PCA():
    def __init__(self, k):
        self.k = k

    def __repr__(self):
        return f'PCA({self.data})'

    @staticmethod
    def Center(data):
        #Convert to torch Tensor and keep the number of rows and columns
        
        if(data.dim() !=3):
            raise IndexError("PCA support just tensor of size: [B, N, M]")

        
        for_subtraction = torch.mean(data, dim = 1)
        
        X = data - for_subtraction.reshape(-1, 1, data.shape[2])
        
        return X
    
    def fit_transform(self, data):
        #Center the Data using the static method within the class
        X = self.Center(data)
        U,S,V = torch.linalg.svd(X, full_matrices=False)

        S = torch.einsum("bi, bij -> bij", S, torch.eye(S.shape[1]).unsqueeze(0).type_as(S))
        
        y = torch.matmul(S, V)
        
        return y
    """
    def fit_transform(self, data):
        #Center the Data using the static method within the class
        X = self.Center(data)
        U,S,V = torch.svd(X)

        S = S[:, :self.k]
        U = U[:,:,:self.k]
        
        y=torch.mul(U, S.unsqueeze(1))
        
        return y
    """
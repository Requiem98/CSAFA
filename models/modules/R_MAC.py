from libraries import *
import utilities as ut
from models.modules.PCA import LearnablePCA


class R_MAC(nn.Module):
    def __init__(self, apply_pca=True, num_comp = None):
        super().__init__()
        self.apply_pca = apply_pca
        
        if(self.apply_pca):
            self.pca = LearnablePCA(512)
        
    def extract_rois(self, x, kernel):
        
        stride = torch.max(torch.tensor(kernel//2), torch.tensor(1)).item()
        # Extract patches
        patches = x.unfold(1, x.shape[1], x.shape[1]).unfold(2, kernel, stride).unfold(3, kernel, stride)
        
        return patches.squeeze(0).reshape(-1, patches.shape[2]*patches.shape[3], patches.shape[4], patches.shape[5], patches.shape[6])
    
    def compute_regions_width(self, l, m):
        return (2*(m/(l+1))).int().item()
     
    def forward(self, x):
        
        b, c, _, _ = x.shape
        
        m = torch.min(torch.tensor(x.size()[2:]))
        
        l1 = torch.amax(self.extract_rois(x, self.compute_regions_width(1, m)), dim = (3,4)) #(B, num_rois, channels)
        l2 = torch.amax(self.extract_rois(x, self.compute_regions_width(2, m)), dim = (3,4)) #(B, num_rois, channels)
        l3 = torch.amax(self.extract_rois(x, self.compute_regions_width(3, m)), dim = (3,4)) #(B, num_rois, channels)
        
        l = torch.cat([l1, l2, l3], dim=1)#(B, num_rois_TOT, channels)
        
        l = f.normalize(l, p=2, dim=2)
        
        if(self.apply_pca):
            l = f.normalize(self.pca(l), p=2, dim=2) #(B, num_rois_TOT, num_comp)
            
        return f.normalize(torch.sum(l, dim = 1), p=2, dim=1) #(B, num_comp)


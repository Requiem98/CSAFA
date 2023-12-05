from libraries import *
import utilities as ut


def compute_recalls(x, y):
    out_euclidean = torch.cdist(x, y)
    
    out_euclidean_ordered = out_euclidean.sort(descending=False)
    
    rk1 = ut.R_K(out_euclidean_ordered[1], k = 1)
    rk5 = ut.R_K(out_euclidean_ordered[1], k = 5)
    rk10 = ut.R_K(out_euclidean_ordered[1], k = 10)
    rk_1_percent = ut.R_K_percent(out_euclidean_ordered[1], k = 1)
    
    return rk1, rk5, rk10, rk_1_percent



def apply_pca(x, y):
        V = torch.pca_lowrank(x, q = 512, center =True)[2]
        x = f.normalize(torch.matmul(x, V), dim=1)
        
        
        V = torch.pca_lowrank(y, q = 512, center =True)[2]
        y = f.normalize(torch.matmul(y, V), dim=1)
        
        return x, y


if __name__ == '__main__':
    Y_ge = torch.tensor(ut.read_object("temp/grd.pkl"))
    gt_Y = torch.tensor(ut.read_object("temp/sat.pkl"))
    
    
    print(Y_ge.shape)
    ###############################################################################
    ##############################TEST DISTANCE####################################
    
    Y_ge, gt_Y = apply_pca(Y_ge, gt_Y)
    
    
    rk1, rk5, rk10, rk_1_percent = compute_recalls(Y_ge, gt_Y)
    
    
    
    print(f"""
          
          ##########Test Retrieval##########
          
          R@1 = {rk1}
          
          R@5 = {rk5}
          
          R@10 = {rk10}
          
          R@1% = {rk_1_percent}
          
          """)


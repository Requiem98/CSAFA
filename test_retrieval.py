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



if __name__ == '__main__':
    Y_ge = torch.tensor(ut.read_object("Data/Y_ge_ModelWrapp.pkl")).cpu()
    gt_Y = torch.tensor(ut.read_object("Data/gt_Y_ModelWrapp.pkl")).cpu()
    
    print(Y_ge.shape)
    ###############################################################################
    ##############################TEST DISTANCE####################################
    
    
    rk1, rk5, rk10, rk_1_percent = compute_recalls(Y_ge, gt_Y)
    
    
    
    print(f"""
          
          ##########Test Retrieval##########
          
          R@1 = {rk1}
          
          R@5 = {rk5}
          
          R@10 = {rk10}
          
          R@1% = {rk_1_percent}
          
          """)


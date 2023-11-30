from libraries import *
import utilities as ut


def DSM(sat_features, grd_features):
    
    n = grd_features.shape[3] - 1
    x = torch.cat([sat_features, sat_features[:, :, :, :n]], dim=3)

    out = f.conv2d(x, grd_features, padding="valid")

    out = out.squeeze()  # shape = [batch_sat, batch_grd, w]
    orien = torch.argmax(out, dim=2)  # shape = [batch_sat, batch_grd]
    
    
    batch_sat, batch_grd = orien.shape
    channel, h, w = sat_features.shape[1:]

    sat_features = sat_features.unsqueeze(1).expand(-1, batch_grd, -1, -1, -1)
    sat_features = sat_features.permute(0, 1, 4, 3, 2)

    orien = orien.unsqueeze(-1)

    i = torch.arange(batch_sat).type_as(orien)
    j = torch.arange(batch_grd).type_as(orien)
    k = torch.arange(w).type_as(orien)
    x, y, z = torch.meshgrid(i, j, k)
    
    x = x.type_as(orien)
    y = y.type_as(orien)
    z = z.type_as(orien)

    z_index = (z + orien) % w
    index = torch.stack([x.reshape(-1), y.reshape(-1), z_index.reshape(-1)], dim=1)


    sat_shifted = sat_features[torch.unbind(index, dim=1)].reshape(batch_sat, batch_grd, w, h, channel)
    
    sat_shifted = sat_features.permute(0, 1, 4, 3, 2)
    
    sat_shifted = f.normalize(sat_shifted, dim=(2,3,4))
    grd_features = f.normalize(grd_features, dim=(1,2,3))
    
    

    return sat_shifted, grd_features

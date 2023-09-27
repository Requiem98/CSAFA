
#########################################################################################
############################ Mixed perspective-polar mapping ############################
#########################################################################################
from libraries import *
import utilities as ut

def sample_within_bounds(signal, x, y, bounds):
    xmin, xmax, ymin, ymax = bounds

    idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)

    sample = np.zeros((x.shape[0], x.shape[1], signal.shape[-1]))
    sample[idxs, :] = signal[x[idxs], y[idxs], :]

    return sample


def sample_bilinear(signal, rx, ry):
    signal_dim_x = signal.shape[0]
    signal_dim_y = signal.shape[1]

    # obtain four sample coordinates
    ix0 = rx.astype(int)
    iy0 = ry.astype(int)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    bounds = (0, signal_dim_x, 0, signal_dim_y)

    # sample signal at each four positions
    signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
    signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
    signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
    signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)

    na = np.newaxis
    # linear interpolation in x-direction
    fx1 = (ix1 - rx)[..., na] * signal_00 + (rx - ix0)[..., na] * signal_10
    fx2 = (ix1 - rx)[..., na] * signal_01 + (rx - ix0)[..., na] * signal_11

    # linear interpolation in y-direction
    return (iy1 - ry)[..., na] * fx1 + (ry - iy0)[..., na] * fx2


############################ Apply Polar Transform to Aerial Images in CVUSA Dataset #############################
S = 750 #800 #750  # Original size of the aerial image
height = 256#800#256#112  # Height of polar transformed aerial image
width = 1024#800#512#616  # Width of polar transformed aerial image

i = np.arange(0, height)
j = np.arange(0, width)
jj, ii = np.meshgrid(j, i)

squareOFmapping = np.array([[0,0], [375-50, S], [S,0], [375+50, S]] ,np.float32)
squareOFmapping2 = np.array([[0,0], [0, S], [S,0], [S, S]] ,np.float32)

#matrix = cv2.getPerspectiveTransform([[0,0], [0, S], [S,0], [S, S]], [[0,0], [0, S], [S,0], [S, S]])
#result = cv2.warpPerspective(frame, matrix, (S, S))

y = S / 2. - S / 2. / height * (height - 1 - ii) * np.sin(2 * np.pi * jj / width)
x = S / 2. + S / 2. / height * (height - 1 - ii) * np.cos(2 * np.pi * jj / width)




#dataset = ut.get_test_dataframe_small()


#images = dataset["aerial_filename"].array


"""
for img in tqdm(images):
    signal = imread("Data/small_CVUSA/"+img)
    image = sample_bilinear(signal, x, y).astype(dtype=np.uint8)
    
    #img_path = img.replace('streetview_aerial', 'streetview_aerial_transformed')
    img_path = "Data/small_CVUSA/"+img.replace('bingmap', 'bingmap_transformed_test')
    dir_path = "/".join(img_path.split("/")[:-1])
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    imsave(img_path.replace("jpg", "png"), image)
"""  

image = imread('./Data/small_CVUSA/bingmap/19/0000001.jpg')


"""
cv2.circle(image, [0,0], 5, [0,0,255])
cv2.circle(image, [375-50, S], 5, [0,0,255])
cv2.circle(image, [S,0], 5, [0,0,255])
cv2.circle(image, [375+50, S], 5, [0,0,255])
"""
matrix = cv2.getPerspectiveTransform(squareOFmapping,squareOFmapping2)
image = cv2.warpPerspective(image, matrix, (S, S))

image = sample_bilinear(image, x, y).astype(dtype=np.uint8)

imsave('./temp/transformed.png', image)



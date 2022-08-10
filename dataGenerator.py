# custom data generator
# without data sugmentation

import os
import numpy as np

# load images
def load_img(img_dir, img_list):
    
    images = []
    
    for i, image_name in enumerate(img_list):
        
        # could add pre-process here 
        
        if (image_name.split('.')[1] == 'npy'):
            
            image = np.load(img_dir + image_name)  # load npy if data type is correct
            images.append(image)
        else:
            print('illegal data format')
            
    images = np.array(images)  # convert into array
    
    return images

def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
    L = len(img_list)
    
    # keras require generator to be infinite, so we use while true
    while True:
        
        batch_start = 0
        batch_end = batch_size
        
        while batch_start < L:
            
            limit = min(batch_end, L) # 考虑最后一个batch分割不完整的情况
            
            # X = load_img(img_list[batch_start:limit])
            # Y = load_img(mask_list[batch_start:limit])
            
            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])
            
            yield(X,Y) # output the X and Y in batch size
            
            batch_start += batch_size # 都往后挪一个batch
            batch_end += batch_size
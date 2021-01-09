from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import random
import os
from PIL import Image
from skimage.transform import resize
import multiprocessing as mp
from multiprocessing import Pool
import itertools

def get_images_wrapper(args):
    return get_images(*args)

# function to retrieve images of a given category and store them as numpy array
def get_images(category, coco, n_images = 1000, offset =  0):
    print("Starting to process {}...".format(category))
    images_dir = './data/{}'.format(category)
    
    # create a directory corresponding to a category
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    
    # get all image ids and corresponding images containing given category
    catIds = coco.getCatIds(catNms = [category])
    imgIds = coco.getImgIds(catIds = catIds)

    # randomize the imgIds list
    random.shuffle(imgIds)

    # count to keep track of the number of valid images generated
    cnt = 0
    
    # list to store all images as numpy arrays
    X = []
    for imgId in imgIds:
        img = coco.loadImgs([imgId])[0]
        
        # use url to load image
        I = io.imread(img['coco_url'])
        
        # load annotations
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        
        # iterate over all annotations and extract bounding boxes
        for ann in anns:
            # if we have got the required number of images, we exit
            if cnt > n_images:
                break
            elif cnt % 100 == 0:
                print("Processed", cnt, "images for", category)

            # Check if entity under consideration meets minimum area criterion
            area = ann['area']
            if area > 0.005*I.shape[0]*I.shape[1]:
                bbox = ann['bbox']
                width = bbox[2]
                height = bbox[3]

                # check if bounding box satisfies the aspect ratio
                if 3/4 <= width/height <= 4/3:
                    # extract bounding box from the original image
                    im = Image.fromarray(I[int(bbox[1]) : int(bbox[1]) + int(height), int(bbox[0]) : int(bbox[0]) + int(width)])

                    # resize the image so that all images have the same size after bounding box removal
                    im = im.resize((520, 520))

                    # add to the list of images
                    X.append(np.array(im))

                    # save the image in the corresponding directory
                    im.save(os.path.join(images_dir, '{}_{}.png'.format(category, cnt + offset)))

                    # increment the count of valid images
                    cnt += 1

    X = np.array(X)
    # print a message if enough data images could not be generated
    if cnt < n_images:
        print("Could not generate enough images for the category => ", category)
    else:
        np.save('{}.npy'.format(os.path.join(images_dir, category)), X)
        print("Saving dataset for category => ", category, X.shape)

if __name__ == "__main__":
    dataDir='.'
    dataType='train2017'
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

    # initialize COCO api for instance annotations
    coco=COCO(annFile)

    args = [('bed', coco, 210, 791), ('car', coco, 436, 565), ('cell phone', coco, 692, 309),
    ('chair', coco, 155, 846), ('microwave', coco, 385, 616), ('person', coco, 279, 722),
    ('refrigerator', coco, 475, 526), ('sink', coco, 1000, 1), ('umbrella', coco, 1000, 1) 
    ]
    with Pool(8) as p:
        #p.map(get_images_wrapper, itertools.product(categories, [coco]))
        p.map(get_images_wrapper, args)
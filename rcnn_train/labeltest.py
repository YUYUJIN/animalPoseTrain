from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import image
from random import sample
from os import path
import json

data_path='D:/test/'

# Plot the keypoints and bounding box for some images
for i in sample(range(1,30),5):

    # Read in keypoints data
    with open(f'{data_path}/annotations.json') as f:
        img_info = json.load(f)

    # Get an image
    img_path = f'{data_path}/train/images/' + str(img_info['images'][str(i)])
    # check if exists otherwise it's in test set
    if path.exists(img_path):
        test_img = image.imread(img_path)
        json_name=img_path.split('/')[-1].replace('.jpg','.json')
        
        # Get the keypoints for the image
        key_data_path = f'{data_path}/train/labels/' + json_name
        with open(key_data_path, 'r') as f:
            key_data = json.loads(f.read())
    
    else:
        img_path = f'{data_path}/val/images/' + str(img_info['images'][str(i)])

        test_img = image.imread(img_path)
        json_name=img_path.split('/')[-1].replace('.jpg','.json')

        # Get the keypoints for the image
        key_data_path = f'{data_path}/val/labels/' + json_name
        with open(key_data_path, 'r') as f:
            key_data = json.loads(f.read())

    # Create figure and axes
    fig, ax = plt.subplots()

    # Annotate over image
    for object in key_data['keypoints']:
        for point in object:
            if point[2] != 0:
                # do stuff
                plt.plot(point[0], point[1], marker='.', color="white")

    plt.imshow(test_img)

    # Create a Rectangle patch
    rects = []
    for bbox in key_data['bboxes']:
        rects.append(patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='r', facecolor='none'))

    # Add the patch to the Axes
    for rect in rects:
        ax.add_patch(rect)

    plt.show()
    plt.clf()
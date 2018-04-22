import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from matplotlib.patches import Circle, RegularPolygon, Rectangle
from matplotlib.collections import PatchCollection
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import PIL
import sys
import shutil
import os

TRAIN_NUM, VAL_NUM, TEST_NUM = 600, 100, 300
IMAGE_WIDTH, IMAGE_HEIGHT = 512, 512  # in pixel
BBOX_EXTRA_SPACE = 5  # in pixel, extra pixel around tight bounding box.
IMAGE_FORMAT = "jpg"
NUM_OBJECTS = 10
SCALE_RANGE = (0.05, 0.2)
COLOR = (0.5, 0.5, 0.5)  # gray

# fix random seed of numpy and python for reproducible result
SEED = 1024
np.random.seed(SEED)
random.seed(SEED)

def reset_dir():
    """Reset dataset directorys, create or clear dirs.
    """
    if os.path.exists(dataset_root_dir):
        shutil.rmtree(dataset_root_dir)
        
    os.mkdir(dataset_root_dir)
    os.mkdir(os.path.join(dataset_root_dir, "train"))
    os.mkdir(os.path.join(dataset_root_dir, "val"))
    os.mkdir(os.path.join(dataset_root_dir, "test"))
    os.mkdir(anno_dir)


def get_figure():
    """Get new figure and axes of matplotlib.
    """
    plt.figure(figsize=(IMAGE_WIDTH / 100, IMAGE_HEIGHT / 100), dpi=100)
    fig = plt.gcf()
    # get rid of white space around border 
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax = plt.gca() 
    ax.axis('off')
    ax.set_aspect('equal')
    
    return fig, ax

def generate_centers(num_objects, scale_range, gap=0.03):
    """Generate object centers that do not overlap with each other or with border.
    
    Note: "scale" here have different meaning, diameter for circle , exciercle diameter for triangle,
        diagonal line lengh for square.
    """
    
    assert 0 < num_objects <= 500 
    assert isinstance(scale_range, tuple) and 0 < scale_range[0] < 1 and scale_range[0] < scale_range[1] < 1
    assert 0 < gap <= 0.5
    
    init_num = num_objects * 5
    xs, ys = np.random.rand(init_num), np.random.rand(init_num)
    scales = np.random.uniform(low=scale_range[0], high=scale_range[1], size=init_num) 
    centers = []
    
    for x, y, scale in zip(xs, ys, scales):
        min_limit, max_limit = (scale + gap) / 2, 1 - (scale + gap) / 2
        # discart center close to border, make sure no object overlap with border
        if x < min_limit or x > max_limit or y < min_limit or y > max_limit:
            continue
        if len(centers) == 0:
            centers.append((x, y, scale))
        else:
            add_in = True
            for center in centers:
                dis = np.hypot(abs(x - center[0]), abs(y - center[1]))
                # discart center close to any selected center, make sure no object overlap with each other
                if dis < (center[2] + scale) / 2 + gap:
                    add_in = False
                    break
            if add_in:
                centers.append((x, y, scale))   
    centers = random.sample(centers, min(num_objects, len(centers)))     

    return centers
        
def generate_figs():
    """Generate image and mask figs.
    """
    
    centers = generate_centers(num_objects=NUM_OBJECTS, scale_range=SCALE_RANGE)  
    
    # generate image fig
    image_fig, ax = get_figure()
    patches = []
    obj_classes = []
    for x, y, scale in centers:
        bottom_left = (x - scale / 2, y - scale / 2)
        class_name = random.choice(['c', 's', 't']) # c: circle, s: square, t: triangle
        if class_name == 'c':
            patches.append(Circle((x, y), radius=scale / 2))
        elif class_name == 's':
            square_len = scale * np.sqrt(2) / 2
            patches.append(Rectangle(bottom_left, width=square_len, height=square_len))
        else:
            patches.append(RegularPolygon((x, y), numVertices=3, radius=scale / 2))
        obj_classes.append(class_name)
        
    collection = PatchCollection(patches)
    collection.set_color(COLOR)
    ax.add_collection(collection)
    
    # generate mask figs
    mask_figs = []
    for patch in patches:
        fig, ax = get_figure()
        fig.set_facecolor((0, 0, 0)) # black
        patch.set_color((1, 1, 1))  # white
        ax.add_artist(patch)
        mask_figs.append(fig)
        
    return image_fig, mask_figs, obj_classes

def fig2numpy(fig):
    """Transform matplotlib figure to numpy array.
    """
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    array = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    array = np.reshape(array, (height, width, 3))
    
    return array
    
    
def write_files(image_id, image_fig, mask_figs, stage):
    """Write image and mask files to disk.
    """
    assert stage in ['train', 'val', 'test']
    
    image_dir = os.path.join(dataset_root_dir, stage, image_id)
    mask_dir = os.path.join(image_dir, "masks")
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
        os.mkdir(mask_dir)
        
    img_path = os.path.join(image_dir, image_id + ".jpg")
    with open(img_path, 'wb') as f:
        image_fig.savefig(f, format=IMAGE_FORMAT)
        plt.close(image_fig)
        
    mask_paths = []    
    for idx, fig in enumerate(mask_figs):
        mask_path = os.path.join(mask_dir, "{}-{}.jpg".format(image_id, idx))
        with open(mask_path, 'wb') as f:
            fig.savefig(f, format=IMAGE_FORMAT, facecolor=(0, 0, 0))
            plt.close(fig)
        mask_paths.append(mask_path)
    
    return img_path, mask_paths

def get_bbox(mask_paths):
    """Get bounding box annotation of masks.
    """
    bboxes = []
    for path in mask_paths:
        mask = PIL.Image.open(path)
        # some pixls around mask have random small value, replace them with 0.
        mask_array = np.array(mask)
        mask_array[mask_array < 50] = 0
        mask_array[mask_array >= 50] = 255
        mask = PIL.Image.fromarray(mask_array)
        bbox = np.array(mask.getbbox())
        # make little bit space around mask
        bbox += np.array([-BBOX_EXTRA_SPACE, -BBOX_EXTRA_SPACE, BBOX_EXTRA_SPACE, BBOX_EXTRA_SPACE])
        bboxes.append(" ".join([str(i) for i in bbox]))
        
    return bboxes

def generate(stage, num_image):
    """Entry function to generate cst-dataset.
    """
    assert stage in ['train', 'val', 'test']
    print("Generating cst {} stage dataset...".format(stage))
    global img_count
    anno_image_ids = []
    anno_object_ids= []
    anno_classes = []
    anno_bboxes = []
    anno_mask_paths = []
    
    for _ in tqdm(range(num_image)):
        image_fig, mask_figs, classes = generate_figs()
        image_id = "cst-{}".format(img_count)
        _, mask_paths = write_files(image_id, image_fig, mask_figs, stage)
        object_ids = ["cst-{}-{}".format(img_count, i) for i in range(len(mask_paths))]
        bboxes = get_bbox(mask_paths)
        
        anno_image_ids.extend([image_id for _ in range(len(mask_figs))])
        anno_object_ids.extend(object_ids)
        anno_classes.extend(classes)
        anno_bboxes.extend(bboxes)
        path_prefix = os.path.join(dataset_root_dir, stage)
        anno_mask_paths.extend([os.path.relpath(path, path_prefix) for path in mask_paths])
        img_count += 1
    
    # write annotation file
    df = pd.DataFrame()
    df['image_id'] = anno_image_ids
    df['object_id'] = anno_object_ids
    df['class'] = anno_classes
    df['bbox'] = anno_bboxes
    df['mask_path'] = anno_mask_paths
    df.to_csv(os.path.join(anno_dir, "anno-{}.csv".format(stage)), index=False)
    
    return df

if __name__ == "__main__":
    output_dir = sys.argv[1]
    dataset_root_dir = os.path.join(output_dir, "cst-dataset")
    anno_dir = os.path.join(dataset_root_dir, "annotation")

    reset_dir()
    img_count = 0
    generate('train', TRAIN_NUM)
    generate('val', VAL_NUM)
    generate('test', TEST_NUM)
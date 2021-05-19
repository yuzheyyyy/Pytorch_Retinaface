import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import random
import numpy as np


cv2.setNumThreads(0)

class WiderFaceDetection(data.Dataset):
    def __init__(self, txt_path, own_txt_path, pattern, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        self.pattern = pattern
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        # for own dataset
        f_own = open(own_txt_path)
        lines_own = f_own.readlines()
        isFirst_own = True
        labels_own = []
        for line in lines_own:
            line = line.rstrip('\n')
            if line.startswith('/opt'):
                if isFirst_own:
                    isFirst_own = False
                else:
                    labels_copy_own = labels_own.copy()
                    self.words.append(labels_copy_own)
                    labels_own.clear()
                self.imgs_path.append(line)
            else:
                line = line.split(' ')[1:]
                label = [float(x) for x in line]
                labels_own.append(label)
        self.words.append(labels_own)


    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):

        img = cv2.imread(self.imgs_path[index])

        height, width, _ = img.shape

        labels = self.words[index]

        # for lable with visible part
        annotations = np.zeros((0, 21))


        if len(labels) == 0:
            return annotations
        
        for idx, label in enumerate(labels):


            '''
            for label with visible
            '''
            annotation = np.zeros((1, 21))

            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[2]  # x2
            annotation[0, 3] = label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[6]    # l1_x
            annotation[0, 7] = label[7]    # l1_y
            annotation[0, 8] = label[12]   # l2_x
            annotation[0, 9] = label[13]   # l2_y
            annotation[0, 10] = label[8]  # l3_x
            annotation[0, 11] = label[9]  # l3_y
            annotation[0, 12] = label[10]  # l4_x
            annotation[0, 13] = label[11]  # l4_y
            annotation[0, 14] = 1
            
            # angle
            annotation[0, 15] = label[14]

            # visible
            annotation[0, 16] = label[15]
            annotation[0, 17] = label[16]
            annotation[0, 18] = label[19]
            annotation[0, 19] = label[17]
            annotation[0, 20] = label[18]
            
            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.pattern == "train":
            if len(labels) >= 2:
                pass
            else:
                rand = np.random.rand()
                if rand < 0.5:
                    rand_idx = random.randint(12880, len(self.imgs_path)-1)
                    rand_img = cv2.imread(self.imgs_path[rand_idx])
                    height_new, width_new, _ = rand_img.shape
                    if height < height_new and width < width_new:
                        pass
                    else:
                        label_rand = self.words[rand_idx]
                        annotations_rand = np.zeros((0, 21))
                        for label in label_rand:
                            annotation_rand = np.zeros((1, 21))

                            # bbox
                            annotation_rand[0, 0] = label[0]  # x1
                            annotation_rand[0, 1] = label[1]  # y1
                            annotation_rand[0, 2] = label[2]  # x2
                            annotation_rand[0, 3] = label[3]  # y2

                            # landmarks
                            annotation_rand[0, 4] = label[4]    # l0_x
                            annotation_rand[0, 5] = label[5]    # l0_y
                            annotation_rand[0, 6] = label[6]    # l1_x
                            annotation_rand[0, 7] = label[7]    # l1_y
                            annotation_rand[0, 8] = label[12]   # l2_x
                            annotation_rand[0, 9] = label[13]   # l2_y
                            annotation_rand[0, 10] = label[8]  # l3_x
                            annotation_rand[0, 11] = label[9]  # l3_y
                            annotation_rand[0, 12] = label[10]  # l4_x
                            annotation_rand[0, 13] = label[11]  # l4_y
                            annotation_rand[0, 14] = 1
                            
                            # angle
                            annotation_rand[0, 15] = label[14]

                            # visible
                            annotation_rand[0, 16] = label[15]
                            annotation_rand[0, 17] = label[16]
                            annotation_rand[0, 18] = label[19]
                            annotation_rand[0, 19] = label[17]
                            annotation_rand[0, 20] = label[18]
                            annotations_rand = np.append(annotations_rand, annotation_rand, axis=0)
                        
                        
                        for i in range(250):
                            resize_ratio = random.uniform(0.4, 0.6)
                            height_rand, width_rand = int(height_new * resize_ratio), int(width_new * resize_ratio)
                            top_left_x = random.randint(0, width)
                            top_left_y = random.randint(0, height)

                            img_box = np.array([top_left_x, top_left_y, top_left_x + width_rand, top_left_y + height_rand])
                            face_box = annotations[0,:4]
                            iou = bb_intersection_over_union(img_box, face_box)
                            if top_left_x + width_rand > width or top_left_y + height_rand > height or iou > 0.02:
                                continue
                            if i < 249:
                                annotations_rand[:, :14] = annotations_rand[:, :14]*resize_ratio + np.tile(np.array([top_left_x, top_left_y]),7).reshape(-1, 14)
                                rand_img_resize = cv2.resize(rand_img, (width_rand, height_rand))
                                img[top_left_y:top_left_y+height_rand, top_left_x:top_left_x+width_rand, :] = rand_img_resize
                                target = np.vstack((target, np.array(annotations_rand)))
                                break


        if self.preproc is not None:
            img, target = self.preproc(img, target, self.imgs_path[index])
        
        return torch.from_numpy(img), target
    
def bb_intersection_over_union(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
from torch.nn.functional import pad
import cv2
import numpy as np
import random

import imgaug as ia
import imgaug.augmenters as iaa
from utils.box_utils import matrix_iof


def _crop(image, boxes, labels, landm, angles, visible, img_dim):
    height, width, _ = image.shape
    pad_image_flag = True

    for _ in range(250):
        """
        if random.uniform(0, 1) <= 0.2:
            scale = 1.0
        else:
            scale = random.uniform(0.3, 1.0)
        """
        PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]
        scale = random.choice(PRE_SCALES)
        short_side = min(width, height)
        w = int(scale * short_side)
        h = w

        if width == w:
            l = 0
        else:
            l = random.randrange(width - w)
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)
        roi = np.array((l, t, l + w, t + h))

        value = matrix_iof(boxes, roi[np.newaxis])
        flag = (value >= 1.0)
        if not flag.any():
            continue

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
        boxes_t = boxes[mask_a].copy()
        labels_t = labels[mask_a].copy()
        landms_t = landm[mask_a].copy()
        landms_t = landms_t.reshape([-1, 5, 2])
        angles_t = angles[mask_a].copy()
        visible_t = visible[mask_a].copy()

        if boxes_t.shape[0] == 0:
            continue

        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] -= roi[:2]
        if np.any(boxes_t<0):
            print('fail')
        # print('box {}'.format(np.any(boxes_t<0)))
        # landm

        
        mask1 = np.any(landms_t < roi[:2], axis=-1)# mask for those invisible landmarks, both original (-1,-1) and (landmark < roi)
        mask2 = np.any(landms_t > roi[2:], axis=-1)
        mask = np.logical_or(mask1, mask2)

        landms_t[:, :, :2] = landms_t[:, :, :2] - roi[:2]
        landms_t[:, :, :2] = np.maximum(landms_t[:, :, :2], np.array([0, 0]))
        landms_t[:, :, :2] = np.minimum(landms_t[:, :, :2], roi[2:] - roi[:2])
        landms_t[mask] = -1
        landms_t = landms_t.reshape([-1, 10])


	# make sure that the cropped image contains at least one face > 16 pixel at training image scale
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
        mask_b = np.minimum(b_w_t, b_h_t) > 0.0
        boxes_t = boxes_t[mask_b]
        labels_t = labels_t[mask_b]
        landms_t = landms_t[mask_b]
        angles_t = angles_t[mask_b]
        visible_t = visible_t[mask_b]

        if boxes_t.shape[0] == 0:
            continue

        pad_image_flag = False

        return image_t, boxes_t, labels_t, landms_t, angles_t, visible_t, pad_image_flag
    return image, boxes, labels, landm, angles, visible, pad_image_flag




def _distort(image):

    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image


def _expand(image, boxes, fill, p):
    if random.randrange(2):
        return image, boxes

    height, width, depth = image.shape

    scale = random.uniform(1, p)
    w = int(scale * width)
    h = int(scale * height)

    left = random.randint(0, w - width)
    top = random.randint(0, h - height)

    boxes_t = boxes.copy()
    boxes_t[:, :2] += (left, top)
    boxes_t[:, 2:] += (left, top)
    expand_image = np.empty(
        (h, w, depth),
        dtype=image.dtype)
    expand_image[:, :] = fill
    expand_image[top:top + height, left:left + width] = image
    image = expand_image

    return image, boxes_t


def _mirror(image, boxes, landms):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]

        # landm
        landms = landms.copy()
        landms = landms.reshape([-1, 5, 2])
        # mask = np.any(landms < 0, axis=-1)
        landms[:, :, 0] = width - landms[:, :, 0]
        # landms[mask] = -1
        tmp = landms[:, 1, :].copy()
        landms[:, 1, :] = landms[:, 0, :]
        landms[:, 0, :] = tmp
        tmp1 = landms[:, 4, :].copy()
        landms[:, 4, :] = landms[:, 3, :]
        landms[:, 3, :] = tmp1
        landms = landms.reshape([-1, 10])

    return image, boxes, landms


def _pad_to_square(image, rgb_mean, pad_image_flag):
    if not pad_image_flag:
        return image
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = rgb_mean
    image_t[0:0 + height, 0:0 + width] = image
    return image_t


def _resize_subtract_mean(image, insize, rgb_mean):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    image -= rgb_mean
    # image /= 128.
    return image.transpose(2, 0, 1)



class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        w, h = img.shape[-2:]

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = mask[np.newaxis, ...].repeat(3, axis=0)
        img = img * mask

        return img


def _transformation(image, boxes, landm):
    h, w, _ = image.shape
    num = np.shape(boxes)[0]
    boxes_all = np.hstack((boxes, boxes[:, [0, 3, 1, 2]]))
    points_all = np.hstack((boxes_all, landm)).reshape(1, -1, 2)


    seq = iaa.Sequential([
        iaa.Affine(
            # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-25, 25), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            # cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode= "constant" # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        ), 
        iaa.PerspectiveTransform(scale=(0.1, 0.15), keep_size=True)])
    
    img_aug, points_aug = seq(image=image, keypoints=points_all)
    points_aug = points_aug.reshape([-1, 2])
    for i in range(num):

        # box
        box1 = np.min(points_aug[9*i:9*i+4, :], axis=0).reshape(2)
        mask1 = (box1 < 0.)
        box1[mask1] = 0.
        box2 = np.max(points_aug[9*i:9*i+4, :], axis=0).reshape(2)
        mask2 = (box2 > np.array([w, h]))
        box2[mask2] = np.array([w, h])[mask2]
        boxes[i, :2] = box1
        boxes[i, 2:] = box2

        # landmarks
        lm = points_aug[9*i+4:9*(i+1), :]
        mask1 = np.any(lm < 0, axis=1)
        mask2 = np.any(lm > np.array([w, h]), axis=1)
        mask = np.logical_or(mask1, mask2)
        lm[mask] = -1
        lm = lm.reshape(10)
        landm[i, :] = lm
    
    return img_aug, boxes, landm


class preproc(object):

    def __init__(self, img_dim, rgb_means):
        self.img_dim = img_dim
        self.rgb_means = rgb_means

    def __call__(self, image, targets, own):
        assert targets.shape[0] > 0, "this image does not have gt"

        '''
        for label with visible
        '''
        boxes = targets[:, :4].copy()
        labels = targets[:, -7].copy()
        landm = targets[:, 4:-7].copy()
        angle = targets[:, -6].copy()
        visible = targets[:, -5:].copy()


        # with visible
        image_t, boxes_t, labels_t, landm_t, angles_t, visible_t, pad_image_flag = _crop(image, boxes, labels, landm, angle, visible, self.img_dim)

        image_t = _distort(image_t)
        image_t = _pad_to_square(image_t,self.rgb_means, pad_image_flag)

        image_t, boxes_t, landm_t = _mirror(image_t, boxes_t, landm_t)      
        height, width, _ = image_t.shape
        # print('self rgb {}'.format(self.rgb_means))
        image_t = _resize_subtract_mean(image_t, self.img_dim, self.rgb_means)
        if own[:4] == '/opt':
            rand = np.random.rand()
            if rand < 0.5:
                cutout = Cutout(n_holes=2, length=40)
                image_t = cutout(image_t)
        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height
        landm_t[:, 0::2] /= width
        landm_t[:, 1::2] /= height

    
        labels_t = np.expand_dims(labels_t, 1)

        '''visible'''
        angles_t = np.expand_dims(angles_t, 1)

        targets_t = np.hstack((boxes_t, landm_t, labels_t, angles_t, visible_t))

        return image_t, targets_t
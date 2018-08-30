"""
Creates an augmented version of the Labeled Faces in the Wild dataset.
Run with:
    python generate_dataset.py --path="/foo/bar/lfw"
"""
from __future__ import print_function, division
import os
import random
import re
import numpy as np
from scipy import misc
from skimage import transform as tf
import argparse

random.seed(43)
np.random.seed(43)

# specs from http://conradsanderson.id.au/lfwcrop/
CROP_UPPER_LEFT_CORNER_X = 83
CROP_UPPER_LEFT_CORNER_Y = 92
CROP_LOWER_RIGHT_CORNER_X = 166
CROP_LOWER_RIGHT_CORNER_Y = 175

WRITE_AUG = True
WRITE_UNAUG = False
SCALE = 64
AUGMENTATIONS = 19

def gen(path, WRITE_AUG_TO, WRITE_UNAUG_TO):
    # """gen method that reads the images, augments and saves them."""
    ds = Dataset(path)
    print("Found %d images total." % (len(ds.fps),))
    
    for img_idx, image in enumerate(ds.get_images()):
        print("Image %d..." % (img_idx,))
        augmentations = augment(image, n=AUGMENTATIONS, hflip=True, vflip=False,
                                scale_to_percent=(0.82, 1.10), scale_axis_equally=True,
                                rotation_deg=8, shear_deg=0,
                                translation_x_px=5, translation_y_px=5,
                                brightness_change=0.1, noise_mean=0.0, noise_std=0.00)
        faces = [image]
        faces.extend(augmentations)
        
        for aug_idx, face in enumerate(faces):
            crop = face[CROP_UPPER_LEFT_CORNER_Y:CROP_LOWER_RIGHT_CORNER_Y+1,
                        CROP_UPPER_LEFT_CORNER_X:CROP_LOWER_RIGHT_CORNER_X+1,
                        ...]
            
            #misc.imshow(face)
            #misc.imshow(crop)
            
            filename = "{:0>6}_{:0>3}.jpg".format(img_idx, aug_idx)
            if WRITE_UNAUG and aug_idx == 0:
                face_scaled = misc.imresize(crop, (SCALE, SCALE))
                misc.imsave(os.path.join(WRITE_UNAUG_TO, filename), face_scaled)
            if WRITE_AUG:
                face_scaled = misc.imresize(crop, (SCALE, SCALE))
                misc.imsave(os.path.join(WRITE_AUG_TO, filename), face_scaled)

    print("Finished.")

def augment(image, n,
            hflip=False, vflip=False, scale_to_percent=1.0, scale_axis_equally=True,
            rotation_deg=0, shear_deg=0, translation_x_px=0, translation_y_px=0,
            brightness_change=0.0, noise_mean=0.0, noise_std=0.0):
    """Augment an image n times.
    Args:
            n                   Number of augmentations to generate.
            hflip               Allow horizontal flipping (yes/no).
            vflip               Allow vertical flipping (yes/no)
            scale_to_percent    How much scaling/zooming to allow. Values are around 1.0.
                                E.g. 1.1 is -10% to +10%
                                E.g. (0.7, 1.05) is -30% to 5%.
            scale_axis_equally  Whether to enforce equal scaling of x and y axis.
            rotation_deg        How much rotation to allow. E.g. 5 is -5 degrees to +5 degrees.
            shear_deg           How much shearing to allow.
            translation_x_px    How many pixels of translation along the x axis to allow.
            translation_y_px    How many pixels of translation along the y axis to allow.
            brightness_change   How much change in brightness to allow. Values are around 0.0.
                                E.g. 0.2 is -20% to +20%.
            noise_mean          Mean value of gaussian noise to add.
            noise_std           Standard deviation of gaussian noise to add.
    Returns:
        List of numpy arrays
    """
    assert n >= 0
    result = []
    if n == 0:
        return result
    
    width = image.shape[0]
    height = image.shape[1]
    matrices = create_aug_matrices(n, img_width_px=width, img_height_px=height,
                                   scale_to_percent=scale_to_percent,
                                   scale_axis_equally=scale_axis_equally,
                                   rotation_deg=rotation_deg,
                                   shear_deg=shear_deg,
                                   translation_x_px=translation_x_px,
                                   translation_y_px=translation_y_px)
    for i in range(n):
        img = np.copy(image)
        matrix = matrices[i]
        
        # random horizontal / vertical flip
        if hflip and random.random() > 0.5:
            img = np.fliplr(img)
        if vflip and random.random() > 0.5:
            img = np.flipud(img)
        
        # random brightness adjustment
        by_percent = random.uniform(1.0 - brightness_change, 1.0 + brightness_change)
        img = img * by_percent
        
        # gaussian noise
        # numpy requires a std above 0
        if noise_std > 0:
            img = img + (255 * np.random.normal(noise_mean, noise_std, (img.shape)))
        
        # clip to 0-255
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        arr = tf.warp(img, matrix, mode="constant") # projects to float 0-1
        img = np.array(arr * 255, dtype=np.uint8)
        result.append(img)
        
    return result

class Dataset(object):
    """Helper class to handle the loading of the LFW dataset dataset."""
    def __init__(self, dirs):
        """Instantiate a dataset object.
        Args:
            dirs    List of filepaths to directories. Direct subdirectories will be read.
        """
        self.dirs = dirs
        self.fps = self.get_filepaths(self.get_direct_subdirectories(dirs))
    
    def get_direct_subdirectories(self, dirs):
        """Find all direct subdirectories of a list of directories.
        Args:
            dirs    List of directories to search in.
        Returns:
            Set of paths to directories
        """
        result = []
        result.extend(dirs)
        for fp_dir in dirs:
            subdirs = [name for name in os.listdir(fp_dir) if os.path.isdir(os.path.join(fp_dir, name))]
            subdirs = [os.path.join(fp_dir, name) for name in subdirs]
            result.extend(subdirs)
        return set(result)
    
    def get_filepaths(self, dirs):
        """Find all jpg-images in provided filepaths.
        Args:
            dirs    List of paths to directories to search in.
        Returns:
            List of filepaths
        """
        result = []
        for fp_dir in dirs:
            fps = [f for f in os.listdir(fp_dir) if os.path.isfile(os.path.join(fp_dir, f))]
            fps = [os.path.join(fp_dir, f) for f in fps]
            fps_img = [fp for fp in fps if re.match(r".*\.jpg$", fp)]
            if len(fps) != len(fps_img):
                print("[Warning] directory '%s' contained %d files with extension differing from 'jpg'" % (fp_dir, len(fps)-len(fps_img)))
            result.extend(fps_img)
        if len(result) < 1:
            print("[Warning] [Dataset] No images of extension *.ppm found in given directories.")
        return result
    
    def get_images(self, start_at=None, count=None):
        """Returns a generator of images.
        Args:
            start_at    Index of first image to return or None.
            count       Maximum number of images to return or None.
        Returns:
            Generator of images (numpy arrays).
        """
        start_at = 0 if start_at is None else start_at
        end_at = len(self.fps) if count is None else start_at+count
        for fp in self.fps[start_at:end_at]:
            image = misc.imread(fp)
            yield image

def create_aug_matrices(nb_matrices, img_width_px, img_height_px,
                        scale_to_percent=1.0, scale_axis_equally=False,
                        rotation_deg=0, shear_deg=0,
                        translation_x_px=0, translation_y_px=0,
                        seed=None):
    """Creates the augmentation matrices that may later be used to transform
    images.

    This is a wrapper around scikit-image's transform.AffineTransform class.
    You can apply those matrices to images using the apply_aug_matrices()
    function.

    Args:
        nb_matrices: How many matrices to return, e.g. 100 returns 100 different
            random-generated matrices (= 100 different transformations).
        img_width_px: Width of the images that will be transformed later
            on (same as the width of each of the matrices).
        img_height_px: Height of the images that will be transformed later
            on (same as the height of each of the matrices).
        scale_to_percent: Same as in ImageAugmenter.__init__().
            Up to which percentage the images may be
            scaled/zoomed. The negative scaling is automatically derived
            from this value. A value of 1.1 allows scaling by any value
            between -10% and +10%. You may set min and max values yourself
            by using a tuple instead, like (1.1, 1.2) to scale between
            +10% and +20%. Default is 1.0 (no scaling).
        scale_axis_equally: Same as in ImageAugmenter.__init__().
            Whether to always scale both axis (x and y)
            in the same way. If set to False, then e.g. the Augmenter
            might scale the x-axis by 20% and the y-axis by -5%.
            Default is False.
        rotation_deg: Same as in ImageAugmenter.__init__().
            By how much the image may be rotated around its
            center (in degrees). The negative rotation will automatically
            be derived from this value. E.g. a value of 20 allows any
            rotation between -20 degrees and +20 degrees. You may set min
            and max values yourself by using a tuple instead, e.g. (5, 20)
            to rotate between +5 und +20 degrees. Default is 0 (no
            rotation).
        shear_deg: Same as in ImageAugmenter.__init__().
            By how much the image may be sheared (in degrees). The
            negative value will automatically be derived from this value.
            E.g. a value of 20 allows any shear between -20 degrees and
            +20 degrees. You may set min and max values yourself by using a
            tuple instead, e.g. (5, 20) to shear between +5 und +20
            degrees. Default is 0 (no shear).
        translation_x_px: Same as in ImageAugmenter.__init__().
            By up to how many pixels the image may be
            translated (moved) on the x-axis. The negative value will
            automatically be derived from this value. E.g. a value of +7
            allows any translation between -7 and +7 pixels on the x-axis.
            You may set min and max values yourself by using a tuple
            instead, e.g. (5, 20) to translate between +5 und +20 pixels.
            Default is 0 (no translation on the x-axis).
        translation_y_px: Same as in ImageAugmenter.__init__().
            See translation_x_px, just for the y-axis.
        seed: Seed to use for python's and numpy's random functions.

    Returns:
        List of augmentation matrices.
    """
    assert nb_matrices > 0
    assert img_width_px > 0
    assert img_height_px > 0
    assert is_minmax_tuple(scale_to_percent) or scale_to_percent >= 1.0
    assert is_minmax_tuple(rotation_deg) or rotation_deg >= 0
    assert is_minmax_tuple(shear_deg) or shear_deg >= 0
    assert is_minmax_tuple(translation_x_px) or translation_x_px >= 0
    assert is_minmax_tuple(translation_y_px) or translation_y_px >= 0

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    result = []

    shift_x = int(img_width_px / 2.0)
    shift_y = int(img_height_px / 2.0)

    # prepare min and max values for
    # scaling/zooming (min/max values)
    if is_minmax_tuple(scale_to_percent):
        scale_x_min = scale_to_percent[0]
        scale_x_max = scale_to_percent[1]
    else:
        scale_x_min = scale_to_percent
        scale_x_max = 1.0 - (scale_to_percent - 1.0)
    assert scale_x_min > 0.0
    #if scale_x_max >= 2.0:
    #     warnings.warn("Scaling by more than 100 percent (%.2f)." % (scale_x_max,))
    scale_y_min = scale_x_min # scale_axis_equally affects the random value generation
    scale_y_max = scale_x_max

    # rotation (min/max values)
    if is_minmax_tuple(rotation_deg):
        rotation_deg_min = rotation_deg[0]
        rotation_deg_max = rotation_deg[1]
    else:
        rotation_deg_min = (-1) * int(rotation_deg)
        rotation_deg_max = int(rotation_deg)

    # shear (min/max values)
    if is_minmax_tuple(shear_deg):
        shear_deg_min = shear_deg[0]
        shear_deg_max = shear_deg[1]
    else:
        shear_deg_min = (-1) * int(shear_deg)
        shear_deg_max = int(shear_deg)

    # translation x-axis (min/max values)
    if is_minmax_tuple(translation_x_px):
        translation_x_px_min = translation_x_px[0]
        translation_x_px_max = translation_x_px[1]
    else:
        translation_x_px_min = (-1) * translation_x_px
        translation_x_px_max = translation_x_px

    # translation y-axis (min/max values)
    if is_minmax_tuple(translation_y_px):
        translation_y_px_min = translation_y_px[0]
        translation_y_px_max = translation_y_px[1]
    else:
        translation_y_px_min = (-1) * translation_y_px
        translation_y_px_max = translation_y_px

    # create nb_matrices randomized affine transformation matrices
    for _ in range(nb_matrices):
        # generate random values for scaling, rotation, shear, translation
        scale_x = random.uniform(scale_x_min, scale_x_max)
        if scale_axis_equally:
            scale_y = scale_x
        else:
            scale_y = random.uniform(scale_y_min, scale_y_max)
        rotation = np.deg2rad(random.randint(rotation_deg_min, rotation_deg_max))
        shear = np.deg2rad(random.randint(shear_deg_min, shear_deg_max))
        translation_x = random.randint(translation_x_px_min, translation_x_px_max)
        translation_y = random.randint(translation_y_px_min, translation_y_px_max)

        # create three affine transformation matrices
        # 1st one moves the image to the top left, 2nd one transforms it, 3rd one
        # moves it back to the center.
        # The movement is neccessary, because rotation is applied to the top left
        # and not to the image's center (same for scaling and shear).
        #print("scale_x", scale_x, "scale_y", scale_y)
        #scale_shift_x = ((img_width_px * scale_x) - img_width_px) / 2
        #scale_shift_y = ((img_height_px * scale_y) - img_height_px) / 2
        #print("img_width_px", img_width_px, "img_height_px", img_height_px)
        #print("scale shift x", scale_shift_x, "scale shift y", scale_shift_y)
        #matrix_scale = tf.AffineTransform(scale=(scale_x, scale_y))
        #matrix_scale_shift = tf.SimilarityTransform(translation=[scale_shift_x, scale_shift_y])
        matrix_to_topleft = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
        matrix_transforms = tf.AffineTransform(scale=(scale_x, scale_y),
                                               rotation=rotation, shear=shear,
                                               translation=(translation_x,
                                                            translation_y))
        matrix_to_center = tf.SimilarityTransform(translation=[shift_x, shift_y])

        # Combine the three matrices to one affine transformation (one matrix)
        #matrix = matrix_scale + matrix_scale_shift + matrix_to_topleft + matrix_transforms + matrix_to_center
        matrix = matrix_to_topleft + matrix_transforms + matrix_to_center

        # one matrix is ready, add it to the result
        result.append(matrix.inverse)

    return result

# if __name__ == "__main__":
    # main()

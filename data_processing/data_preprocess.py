# preprocessing steps in config.yaml
# cutting image up and converting label coordinates

import ujson as json
import os
import numpy as np
import cv2
from PIL import Image
import gc
import time


def get_json_dict(filepath):
    with open(filepath) as json_file:
        anno = json.load(json_file)
    return anno


def get_json_img_list(json_dict):
    img_list = []
    for img in json_dict['images']:
        img_list.append(img['file_name'])
    return img_list


def compute_polygon_area(segmentation):
    # Assuming segmentation is a list of lists, where each list is a flat array of x, y coordinates
    # Example: segmentation = [[x1, y1, x2, y2, ..., xn, yn]]
    poly = np.array(segmentation).reshape(-1, 2)
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def is_anno_out_of_bounds(annotation):
    """
    Checks if an annotation (either bbox or segmentation) is completely out of bounds.
    Returns True if all points are negative.
    """
    oob = False

    # if 'bbox' in annotation:
    #     # bbox in [x, y, width, height] format
    #     x, y, width, height = annotation['bbox']
    #     if (x + width <= 0) or (y + height <= 0) or (x < 0 and y < 0):
    #         oob = True

    if 'segmentation' in annotation and annotation['segmentation']:
        # Assume segmentation is a list of polygons, where each is a list of [x1, y1, x2, y2, ..., xn, yn]
        for polygon in annotation['segmentation']:
            # Check if all coordinates in the polygon are negative
            points = np.array(polygon).reshape(-1, 2)
            if np.all(points < 0):
                oob = True

    return oob


def remove_out_of_bounds_annotations(annotations):
    return [ann for ann in annotations if not is_anno_out_of_bounds(ann)]


def derive_bbox_from_segmentation(segmentation):
    """
    Derives a bounding box from segmentation coordinates.
    Assumes segmentation is a list of lists, each containing [x1, y1, x2, y2, ..., xn, yn].
    Returns [x, y, width, height].
    """
    # Flatten the list of points and convert to a numpy array directly
    coords = np.concatenate([np.array(seg).reshape(-1, 2) for seg in segmentation])

    # Calculate bounding box directly
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    width = x_max - x_min
    height = y_max - y_min

    return [x_min, y_min, width, height]


def grayscale_image(image_dir, output_dir, image_filename):
    img_path = os.path.join(image_dir, image_filename)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Convert back to 3-channel
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(os.path.join(output_dir, image_filename), gray_bgr)


# def is_image_black(image_path):
#     """
#     Check if an image is completely black (all pixel values are 0).
#     """
#     image = cv2.imread(image_path)
#     if image is None:
#         return False  # Ignore if the image could not be read (corrupt or invalid format)
#     return not image.any()  # True if all pixels are 0


def slice_image(image_dir, image_filename, output_dir, slice_size, img_id=None, save_images=True,
                save_dict=False) -> list:
    # returns list of dicts
    image_path = os.path.join(image_dir, image_filename)

    img = Image.open(image_path)
    img_width, img_height = img.size

    slice_width, slice_height = slice_size

    assert slice_width < img_width
    assert slice_height < img_height

    # Calculate number of slices
    x_slices = img_width // slice_width
    y_slices = img_height // slice_height

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    slice_info = []
    slice_idx = 0
    for i in range(x_slices):
        for j in range(y_slices):
            x = i * slice_width  # x is x-coord of top left corner
            y = j * slice_height  # y is y-coord of top left corner
            w = slice_width
            h = slice_height

            # Crop the image
            box = (x, y, x + w, y + h)
            img_slice = img.crop(box)

            # Skip if image all black
            arr = np.asarray(img_slice)
            extrema = (arr.min(), arr.max())
            if not extrema == (0, 0):
                # Save subimage
                local_filename = f"{image_filename}_slice_{slice_idx}.jpg"
                slice_filename = os.path.join(output_dir, local_filename)
                if save_images:
                    img_slice.save(slice_filename)

                # Generate output dict
                slice_info.append({
                    "file_name": local_filename,
                    "height": slice_height,
                    "width": slice_width,
                    "orig_id": img_id,
                    "slice_idx": slice_idx,
                    "id": img_id * 1000 + slice_idx,
                    "xywh": box})
                slice_idx += 1

    if save_dict:
        with open(os.path.join(os.path.dirname(output_dir), "slice_info.json"), "w") as outfile:
            json.dump(slice_info, outfile)

    return slice_info


def adjust_coco_annotations(orig_img_filename, coco_annots, slice_info, orig_img_id=None) -> list:
    adjusted_annotations = []

    if isinstance(coco_annots, str):
        with open(coco_annots) as json_file:
            orig_json = json.load(json_file)
    else:
        orig_json = coco_annots

    if not orig_img_id:
        orig_img_id = [b['id'] for b in orig_json['images'] if b.get('file_name') == orig_img_filename][0]

    orig_img_annots = [b for b in orig_json['annotations'] if b.get('image_id') == orig_img_id]

    for slice_idx, slice_data in enumerate(slice_info):  # slice_idx is the slice number
        if slice_idx % 50 == 0:
            print(f'slice index: {slice_idx} of {len(slice_info)}')

        slice_x, slice_y = slice_data['xywh'][:2]  # Top-left of the slice

        for annots in orig_img_annots:
            adjusted_annot = annots.copy()

            # Adjust segmentation points
            new_segmentation = []
            for seg in annots['segmentation']:
                new_seg = []
                for i in range(0, len(seg), 2):
                    x = seg[i] - slice_x
                    y = seg[i + 1] - slice_y
                    new_seg.append(x)
                    new_seg.append(y)

                if max(new_seg) > 0:
                    new_segmentation.append(new_seg)

                if len(new_segmentation) > 0:
                    adjusted_annot['orig_id'] = annots['id']
                    adjusted_annot['id'] = (slice_idx + 1) * 100000 + annots['id']
                    adjusted_annot['orig_image_id'] = annots['image_id']
                    adjusted_annot['image_id'] = slice_data['id']

                    adjusted_annot['bbox'] = derive_bbox_from_segmentation(new_segmentation)
                    adjusted_annot['segmentation'] = new_segmentation
                    adjusted_annot['area'] = compute_polygon_area(new_segmentation)

                    adjusted_annotations.append(adjusted_annot)

    return adjusted_annotations


def slice_batch(img_dir, json_path, size, output_dir, batch_size=None, save_images=True, save_dict=False):
    orig_json = get_json_dict(json_path)
    orig_img_list = get_json_img_list(orig_json)
    if batch_size is not None:
        batch_size = int(batch_size)
        orig_img_list = orig_img_list[0:batch_size]

    subimages = []
    subannots = []

    print(f'images to slice: {len(orig_img_list)}')
    i = 1
    for img_path in orig_img_list:
        gc.collect()
        print(f'image number: {i}')
        orig_img_id = [b['id'] for b in orig_json['images'] if b.get('file_name') == img_path][0]
        subimage_info = slice_image(image_dir=img_dir,
                                    image_filename=img_path,
                                    output_dir=os.path.join(output_dir, 'images'),
                                    slice_size=size,
                                    img_id=orig_img_id,
                                    save_images=save_images,
                                    save_dict=save_dict)
        subimages += subimage_info

        subimage_annot = adjust_coco_annotations(orig_img_filename=img_path,
                                                 coco_annots=orig_json,
                                                 slice_info=subimage_info,
                                                 orig_img_id=orig_img_id)
        subannots += subimage_annot
        i += 1

    new_json = {'info': orig_json['info'],
                'images': subimages,
                'annotations': subannots,
                'categories': orig_json['categories']}
    print(f'length images: {len(subimages)}')

    gc.collect()

    print(f'saving json')
    with open(os.path.join(output_dir, "labels.json"), "w") as outfile:
        json.dump(new_json, outfile)

    return new_json

# class Dataset():
#     def __init__(self):

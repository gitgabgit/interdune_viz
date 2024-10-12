# preprocessing steps in config.yaml
# cutting image up and converting label coordinates

import json
import os
import numpy as np
from PIL import Image


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


def grayscale_image(image_dir, output_dir, image_filename):
    img = Image.open(os.path.join(image_dir, image_filename)).convert("L")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img.save(os.path.join(output_dir, image_filename))


def slice_image(image_dir, image_filename, output_dir, slice_size, img_id=None) -> list:
    # returns list of dicts
    image_path = os.path.join(image_dir, image_filename)

    img = Image.open(image_path)
    img_width, img_height = img.size

    slice_width, slice_height = slice_size

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

            # Save subimage
            local_filename = f"{image_filename}_slice_{slice_idx}.jpg"
            slice_filename = os.path.join(output_dir, local_filename)
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

    return slice_info


def adjust_coco_annotations(orig_img_filename, coco_annots, slice_info, orig_img_id=None) -> list:
    # returns list of dicts
    adjusted_annotations = []

    # if loading json from filename
    # with open(coco_annotations) as json_file:
    #     orig_json = json.load(json_file)

    # if loading json dict
    orig_json = coco_annots

    if not orig_img_id:
        orig_img_id = [b['id'] for b in orig_json['images'] if b.get('file_name') == orig_img_filename][0]

    orig_img_annots = [b for b in orig_json['annotations'] if b.get('image_id') == orig_img_id]

    for slice_idx, slice_data in enumerate(slice_info):
        slice_x, slice_y = slice_data['xywh'][:2]  # Top-left of the slice

        annot_id = 1
        for annots in orig_img_annots:

            adjusted_annot = annots.copy()

            # Adjust the bounding box
            bbox = annots['bbox']
            bbox[0] -= slice_x  # Shift x-coordinate
            bbox[1] -= slice_y  # Shift y-coordinate

            # Adjust segmentation points
            new_segmentation = []
            for seg in annots['segmentation']:
                new_seg = []
                for i in range(0, len(seg), 2):
                    x = seg[i] - slice_x
                    y = seg[i + 1] - slice_y
                    new_seg.append(x)
                    new_seg.append(y)
                new_segmentation.append(new_seg)

            adjusted_annot['orig_id'] = annots['id']
            adjusted_annot['id'] = slice_data['id'] * 100 + annot_id  # how to do a new?
            adjusted_annot['orig_image_id'] = annots['image_id']
            adjusted_annot['image_id'] = slice_data['id']  # slice info id for appropriate image??

            adjusted_annot['bbox'] = bbox
            adjusted_annot['segmentation'] = new_segmentation
            adjusted_annot['area'] = compute_polygon_area(new_segmentation)

            # adjusted_annot['iscrowd'] = annots['iscrowd']
            # adjusted_annot['category_id'] = annots['category_id']
            # adjusted_annot['extra'] = annots['extra']

            adjusted_annotations.append(adjusted_annot)
            annot_id += 1

    return adjusted_annotations


def slice_batch(img_dir, json_path, size, output_dir):
    orig_json = get_json_dict(json_path)
    orig_img_list = get_json_img_list(orig_json)

    subimages = []
    subannots = []

    for img_path in orig_img_list:
        # full_path = os.path.join(img_dir, img_path)
        orig_img_id = [b['id'] for b in orig_json['images'] if b.get('file_name') == img_path][0]
        subimage_info = slice_image(image_dir=img_dir,
                                    image_filename=img_path,
                                    output_dir=os.path.join(output_dir, 'images'),
                                    slice_size=size,
                                    img_id=orig_img_id)
        subimages += subimage_info

        subimage_annot = adjust_coco_annotations(orig_img_filename=img_path,
                                                 coco_annots=orig_json,
                                                 slice_info=subimage_info,
                                                 orig_img_id=orig_img_id)
        subannots += subimage_annot

    new_json = {'info': orig_json['info'],
                'images': subimages,
                'annotations': subannots,
                'categories': orig_json['categories']}

    with open(os.path.join(output_dir, "labels.json"), "w") as outfile:
        json.dump(new_json, outfile)

    return new_json

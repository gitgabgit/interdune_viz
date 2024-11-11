# load a dataset and associated label sheet
import json
import os
import pandas as pd
import numpy as np

from config_utils import Config
from data_processing.data_preprocess import remove_out_of_bounds_annotations, derive_bbox_from_segmentation


def get_anno_path(anno_dir, date_str: str = ''):
    anno_list = os.listdir(anno_dir)
    if len(anno_list) > 1:
        anno_filepath = os.path.join(anno_dir, date_str)
    else:
        anno_filepath = os.path.join(anno_dir, anno_list[0])
    return anno_filepath


def split_json_train_test_val(json_filepath: str,
                              image_dirs: list,
                              anno_dirs: list) -> None:

    with open(json_filepath) as json_file:
        labels = json.load(json_file)
        labels_images = pd.DataFrame.from_dict(labels['images'])
        labels_annots = pd.DataFrame.from_dict(labels['annotations'])

        split_sets = ['train', 'val', 'test']

        for i in range(len(split_sets)):
            split_name = split_sets[i]
            files_list = os.listdir(image_dirs[i])

            labels_images[split_name] = np.where(labels_images['file_name'].isin(files_list), 1, 0)
            image_list = labels_images.loc[labels_images[split_name] == 1].copy().drop(
                columns=[split_name]).to_dict(orient='records')

            labels_annots[split_name] = np.where(
                labels_annots['image_id'].isin(labels_images.loc[labels_images[split_name] == 1]['id'].unique()), 1, 0)
            annot_list = labels_annots.loc[labels_annots[split_name] == 1].copy().drop(
                columns=[split_name]).to_dict(orient='records')

            split_dict = {'info': labels['info'],
                          'images': image_list,
                          'annotations': annot_list,
                          'categories': labels['categories']}

            with open(anno_dirs[i], "w") as outfile:
                json.dump(split_dict, outfile)
                outfile.close()

        json_file.close()



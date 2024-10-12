# load a dataset and associated label sheet
import json
import os
import pandas as pd
import numpy as np

from config_utils import Config


def get_anno_path(anno_dir, date_str: str = ''):
    anno_list = os.listdir(anno_dir)
    if len(anno_list) > 1:
        anno_filepath = os.path.join(anno_dir, date_str)
    else:
        anno_filepath = os.path.join(anno_dir, anno_list[0])
    return anno_filepath


def split_json_train_test_val(cfg: dict,
                              json_filepath: str,
                              image_dirs: list,
                              anno_dirs: list) -> None:

                              # train_image_dir: str,
                              # test_image_dir: str,
                              # val_image_dir: str,
                              # train_json_filepath: str,
                              # test_json_filepath: str,
                              # val_json_filepath: str) -> None:

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

        # train_files = os.listdir(train_image_dir)
        # test_files = os.listdir(test_image_dir)
        # val_files = os.listdir(val_image_dir)

        # labels_images['train'] = np.where(labels_images['file_name'].isin(train_files), 1, 0)
        # labels_images['test'] = np.where(labels_images['file_name'].isin(test_files), 1, 0)
        # labels_images['val'] = np.where(labels_images['file_name'].isin(val_files), 1, 0)

        # train_image_list = labels_images.loc[labels_images['train'] == 1].copy().drop(columns=['train', 'test', 'val']).to_dict(
        #     orient='records')
        # test_image_list = labels_images.loc[labels_images['test'] == 1].copy().drop(columns=['train', 'test', 'val']).to_dict(
        #     orient='records')
        # val_image_list = labels_images.loc[labels_images['val'] == 1].copy().drop(columns=['train', 'test', 'val']).to_dict(
        #     orient='records')

        # labels_annots['train'] = np.where(
        #     labels_annots['image_id'].isin(labels_images.loc[labels_images.train == 1]['id'].unique()), 1, 0)
        # labels_annots['val'] = np.where(
        #     labels_annots['image_id'].isin(labels_images.loc[labels_images.val == 1]['id'].unique()), 1, 0)

        # train_annot_list = labels_annots.loc[labels_annots['train'] == 1].copy().drop(columns=['train', 'val']).to_dict(
        #     orient='records')
        # val_annot_list = labels_annots.loc[labels_annots['val'] == 1].copy().drop(columns=['train', 'val']).to_dict(
        #     orient='records')

        # train_dict = {'info': labels['info'],
        #               'images': train_image_list,
        #               'annotations': train_annot_list,
        #               'categories': labels['categories']}
        # val_dict = {'info': labels['info'],
        #             'images': val_image_list,
        #             'annotations': val_annot_list,
        #             'categories': labels['categories']}

        # with open(train_json_filepath, "w") as outfile:
        #     json.dump(train_dict, outfile)
        #     outfile.close()
        #
        # with open(val_json_filepath, "w") as outfile:
        #     json.dump(val_dict, outfile)
        #     outfile.close()



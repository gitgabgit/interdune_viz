import os
import yaml
import pprint


def get_wd() -> str:
    return os.getcwd()


class Config:
    def __init__(self, cfg_path: str = 'config.yaml') -> None:
        self.cfg: dict = yaml.safe_load(open(cfg_path))
        self.env_wd = get_wd()
        self.preprocess_steps = self.cfg['preprocess_steps']
        self.raw_paths = {}
        self.preproc_paths = {}
        self.split_paths = {}

    def get_paths(self):

        if self.env_wd == '/content':
            env = 'google_colab'
        else:  # TODO testing locally change to none
            env = 'google_colab'

        raw_wd: str = self.cfg['env_paths'][env]['working_dir']
        raw_data_dir: str = self.cfg['env_paths'][env]['data_path']
        raw_path__data: str = os.path.join(raw_wd, raw_data_dir)

        output_dir: str = self.cfg['env_paths'][env]['output_path']
        output_model_dir: str = os.path.join(output_dir, 'model')  # TODO output this somewhere

        datasets = self.cfg['datasets']

        raw_paths = {}
        preproc_paths = {}
        split_paths = {}

        # TODO this step should be when the preprocessing actually happens
        dir_suffix = 'preprocess_'
        for i in self.cfg['preprocess_steps']:
            dir_suffix += f'_{i}'

        for dataset in datasets:
            output_data_dir: str = os.path.join(output_dir, 'data', dataset)

            raw_paths[dataset] = {}
            raw_paths[dataset]['top_dir'] = os.path.join(raw_path__data, self.cfg['dataset_paths'][dataset])
            raw_paths[dataset]['images_dir'] = os.path.join(raw_paths[dataset]['top_dir'], 'images')
            raw_paths[dataset]['labels_dir'] = os.path.join(raw_paths[dataset]['top_dir'], 'labels')

            preproc_paths[dataset] = {}
            preproc_paths[dataset]['top_dir'] = os.path.join(output_data_dir, dir_suffix)
            preproc_paths[dataset]['images_dir'] = os.path.join(preproc_paths[dataset]['top_dir'], 'images')
            preproc_paths[dataset]['labels'] = os.path.join(preproc_paths[dataset]['top_dir'], 'labels.json')

            split_paths[dataset] = {}
            split_paths[dataset]['top_dir'] = output_data_dir
            split_paths[dataset]['train_dir'] = os.path.join(output_data_dir, 'train')
            split_paths[dataset]['train_images_dir'] = os.path.join(split_paths[dataset]['train_dir'], 'images')
            split_paths[dataset]['train_labels'] = os.path.join(split_paths[dataset]['train_dir'], 'labels.json')

            split_paths[dataset]['val_dir'] = os.path.join(output_data_dir, 'val')
            split_paths[dataset]['val_images_dir'] = os.path.join(split_paths[dataset]['val_dir'], 'images')
            split_paths[dataset]['val_labels'] = os.path.join(split_paths[dataset]['val_dir'], 'labels.json')

            split_paths[dataset]['test_dir'] = os.path.join(output_data_dir, 'test')
            split_paths[dataset]['test_images_dir'] = os.path.join(split_paths[dataset]['test_dir'], 'images')
            split_paths[dataset]['test_labels'] = os.path.join(split_paths[dataset]['test_dir'], 'labels.json')

        self.raw_paths = raw_paths
        self.preproc_paths = preproc_paths
        self.split_paths = split_paths


if __name__ == '__main__':

    c = Config()
    c.get_paths()

# # # TODO grab most recent json labels file


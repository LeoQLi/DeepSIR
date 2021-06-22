import os, sys
import logging
import torchvision

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
import dataloader.transformation as Transforms
from dataloader.oxford_loader import Oxford
from dataloader.threeDMatch_loader import ThreeDMatch
from dataloader.kitti_loader import KITTIPair, SemanticKITTIPair

_logger = logging.getLogger('Data')


def get_train_datasets(args):
    # data transformation
    train_transforms, val_transforms = get_transforms(noise_type=args.noise_type,
                                                    rot_mag=args.rot_mag, trans_mag=args.trans_mag,
                                                    num_points=args.num_points, xy_rot_scale=args.xy_rot_scale,
                                                    partial_p_keep=args.crop_partial)
    _logger.info('Train transforms: {}'.format(', '.join([type(t).__name__ for t in train_transforms])))
    _logger.info('Val transforms: {}'.format(', '.join([type(t).__name__ for t in val_transforms])))
    train_transforms = torchvision.transforms.Compose(train_transforms)
    val_transforms = torchvision.transforms.Compose(val_transforms)

    # data object
    args.dataset_path = os.path.join(args.dataset_path, args.dataset_type)

    if args.dataset_type == 'Modelnet':
        train_categoryfile = './dataloader/modelnet40_half1.txt'
        val_categoryfile = './dataloader/modelnet40_half1.txt'
        train_categories, val_categories = None, None

        train_categories = [line.rstrip('\n') for line in open(train_categoryfile)]
        train_categories.sort()

        val_categories = [line.rstrip('\n') for line in open(val_categoryfile)]
        val_categories.sort()

        train_data = ModelNet(args.dataset_path, subset='train', categories=train_categories, transform=train_transforms)
        val_data   = ModelNet(args.dataset_path, subset='test', categories=val_categories, transform=val_transforms)

    elif args.dataset_type == 'Oxford':
        train_data = OxfordSingleFrame(args, mode='train', transform=train_transforms)
        val_data   = OxfordSingleFrame(args, mode='val', transform=val_transforms)

    elif args.dataset_type == 'KITTI':
        train_data = KITTISingleFrame(args, mode='train', transform=train_transforms)
        val_data   = KITTISingleFrame(args, mode='val', transform=val_transforms)
    else:
        raise NotImplementedError

    return train_data, val_data


def get_test_datasets(args):
    _, test_transforms = get_transforms(noise_type=args.noise_type,
                                        rot_mag=args.rot_mag, trans_mag=args.trans_mag,
                                        num_points=args.num_points, xy_rot_scale=args.xy_rot_scale,
                                        partial_p_keep=args.crop_partial)
    _logger.info('Test transforms: {}'.format(', '.join([type(t).__name__ for t in test_transforms])))
    test_transforms = torchvision.transforms.Compose(test_transforms)

    # data object
    args.dataset_path = os.path.join(args.dataset_path, args.dataset_type)

    if args.dataset_type == 'Modelnet':
        test_category_file = './data_loader/modelnet40_half2.txt'
        test_categories = None
        test_categories = [line.rstrip('\n') for line in open(test_category_file)]
        test_categories.sort()

        test_data = ModelNet(args.dataset_path, subset='test', categories=test_categories, transform=test_transforms)

    elif args.dataset_type == 'Oxford':
        test_data = OxfordSingleFrame(args, mode='test', transform=test_transforms)

    elif args.dataset_type == 'KITTI':
        test_data = KITTISingleFrame(args, mode='test', transform=test_transforms)
    else:
        raise NotImplementedError

    return test_data


def get_transforms(noise_type, rot_mag=45.0, trans_mag=0.5, num_points=1024, xy_rot_scale=1.0, partial_p_keep=None):
    """Get the list of transformation to be used for training or evaluating

    Args:
        noise_type: Either 'clean', 'jitter', 'crop'.
            Depending on the option, some of the subsequent arguments may be ignored.
        rot_mag: Magnitude of rotation perturbation to apply to source, in degrees.
        trans_mag: Magnitude of translation perturbation to apply to source.
        num_points: Number of points to uniformly resample to.
            Note that this is with respect to the full point cloud. The number of
            points will be proportionally less if cropped
        xy_rot_scale: Discount of rotation in XY axis
        partial_p_keep: Proportion to keep during cropping, [src_p, ref_p]
            Default: [0.7, 0.7], i.e. Crop both source and reference to ~70%

    Returns:
        train_transforms, test_transforms: Both contain list of transformations to be applied
    """
    partial_p_keep = partial_p_keep if partial_p_keep is not None else [0.7, 0.7]

    # 1-1 correspondence for each point (resample first before splitting), no noise
    if noise_type == "clean":
        train_transforms = [Transforms.Resampler(num_points),
                            Transforms.RandomRotatorZ(),
                            Transforms.SplitSourceRef(),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag, xy_rot_scale=xy_rot_scale),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.FixedResampler(num_points),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag, xy_rot_scale=xy_rot_scale),
                           Transforms.ShufflePoints()]

    # Points randomly sampled (might not have perfect correspondence), gaussian noise to position
    elif noise_type == "jitter":
        train_transforms = [Transforms.RandomRotatorZ(),
                            Transforms.SplitSourceRef(),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag, xy_rot_scale=xy_rot_scale),
                            Transforms.Resampler(num_points),
                            Transforms.RandomJitter(),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag, xy_rot_scale=xy_rot_scale),
                           Transforms.Resampler(num_points),
                           Transforms.RandomJitter(),
                           Transforms.ShufflePoints()]

    # Both source and reference point clouds cropped, plus same noise in "jitter"
    elif noise_type == "crop":
        train_transforms = [Transforms.RandomRotatorZ(),
                            Transforms.SplitSourceRef(),
                            Transforms.RandomCrop(partial_p_keep),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag, xy_rot_scale=xy_rot_scale),
                            Transforms.Resampler(num_points),
                            Transforms.RandomJitter(),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomCrop(partial_p_keep),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag, xy_rot_scale=xy_rot_scale),
                           Transforms.Resampler(num_points),
                           Transforms.RandomJitter(),
                           Transforms.ShufflePoints()]
    else:
        raise NotImplementedError

    return train_transforms, test_transforms


def get_train_datasets_V2(args):
    args.dataset_path = os.path.join(args.dataset_path, args.dataset_type)

    if args.dataset_type == 'Oxford':
        # train_transforms, _ = get_transforms(noise_type=args.noise_type,
        #                                 rot_mag=args.rot_mag, trans_mag=args.trans_mag,
        #                                 num_points=args.num_points, xy_rot_scale=args.xy_rot_scale,
        #                                 partial_p_keep=args.crop_partial)
        # _logger.info('Train transforms: {}'.format(', '.join([type(t).__name__ for t in train_transforms])))
        # train_transforms = torchvision.transforms.Compose(train_transforms)

        train_data = Oxford(args, split='train')
        val_data   = Oxford(args, split='val')

    elif args.dataset_type == 'KITTI':
        # train_data = KITTIPair(args, split='train')
        # val_data   = KITTIPair(args, split='val')
        train_data = SemanticKITTIPair(args, split='train')
        val_data   = SemanticKITTIPair(args, split='val')

    elif args.dataset_type == '3DMatch':
        train_data = ThreeDMatch(args, split='train')
        val_data   = ThreeDMatch(args, split='val')

    else:
        raise NotImplementedError

    return train_data, val_data


def get_test_datasets_V2(args):
    args.dataset_path = os.path.join(args.dataset_path, args.dataset_type)

    if args.dataset_type == 'Oxford':
        test_data = Oxford(args, split='test')

    elif args.dataset_type == 'KITTI':
        # test_data = KITTIPair(args, split='test')
        test_data = SemanticKITTIPair(args, split='test')

    elif args.dataset_type == '3DMatch':
        test_data = ThreeDMatch(args, split='test')

    else:
        raise NotImplementedError

    return test_data


### ***************************************************************************
# import h5py
# import yaml
import pickle
import numpy as np
from torch.utils.data import Dataset
from dataloader.data_base import DataBase, process_point_cloud


class OxfordSingleFrame(DataBase):
    def __init__(self, args, mode=None, transform=None):
        super(OxfordSingleFrame, self).__init__(args)
        assert mode in ['train', 'val', 'test']

        self.root_path = args.dataset_path
        self.feat_len = args.feat_len
        self.num_val = args.num_val
        self.mode = mode
        self.transform = transform

        self._logger.info('Loading data from {}'.format(self.root_path))

        if not os.path.exists(os.path.join(self.root_path)):
            self._logger.info('Invalid path: {}'.format(self.root_path))

        if self.mode == 'train':
            self.dataset = self.make_train_dataset()
        else:
            self.dataset = self.make_test_dataset()
            if self.num_val > 0:
                self.dataset = self.dataset[:self.num_val]

        self._logger.info('Found {} {} instances.'.format(len(self.dataset), self.mode))

    def __len__(self):
        """ Return the total number of samples
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """ Generates one sample of data
        """
        if index in self.cache:
            pc = self.cache[index]
        else:
            pc = self.get_data(index)
            if len(self.cache) < self.cache_size:
                self.cache[index] = pc

        data = {'points': pc,
                   'idx': np.array(index, dtype=np.int32)}

        if self.transform:
            data = self.transform(data)

        return data

    def get_data(self, index):
        """ Read data from npy file and pre-process the data
        Return pc_np: 7 channel [x, y, z, nx, ny, nz, curvature]
        """
        if self.mode == 'train':
            filename = self.dataset[index]['file']
            pc_np = np.load(os.path.join(self.root_path, 'train_np_nofilter', filename[0:-3] + 'npy'))
        else:
            anc_idx = self.dataset[index]['anc_idx']
            pc_np = np.load(os.path.join(self.root_path, 'test_models_20k_np_nofilter', '%d.npy' % anc_idx))

        pc_np = process_point_cloud(pc_np, r_min=0.0, r_max=50.0, z_min=-3.0, z_max=20.0, grid_size=0.0)
        return pc_np[:, :self.feat_len]

    def make_train_dataset(self):
        f = open(os.path.join(self.root_path, 'train_relative.txt'), 'r')
        lines_list = f.readlines()

        dataset = []
        for i, line_str in enumerate(lines_list):
            # convert each line to a dict
            line_splitted_list = line_str.split('|')
            try:
                assert len(line_splitted_list) == 3
            except Exception:
                self._logger.info('Invalid line.')
                self._logger.info(i)
                self._logger.info(line_splitted_list)
                continue

            file_name = line_splitted_list[0].strip()
            positive_lines = list(map(int, line_splitted_list[1].split()))
            non_negative_lines = list(map(int, line_splitted_list[2].split()))

            data = {'file': file_name, 'pos_list': positive_lines, 'nonneg_list': non_negative_lines}
            dataset.append(data)
        f.close()
        return dataset  # [{'file', 'pos_list', 'nonneg_list'}]

    def make_test_dataset(self):
        with open(os.path.join(self.root_path, 'test_models_20k_np_nofilter', 'groundtruths.pkl'), 'rb') as f:
            return pickle.load(f)  # [['anc_idx', 'pos_idx', 'neg_idx', 'q', 't']]


class KITTISingleFrame(DataBase):
    def __init__(self, args, mode=None, transform=None):
        super(KITTISingleFrame, self).__init__(args)
        assert mode in ['train', 'val', 'test']

        self.root_path = args.dataset_path
        self.feat_len = args.feat_len
        self.mode = mode
        self.transform = transform

        self._logger.info('Loading data from {}'.format(self.root_path))

        if not os.path.exists(os.path.join(self.root_path)):
            self._logger.info('Invalid path: {}'.format(self.root_path))

        self.seq_list, self.folder_list, self.sample_num_list, self.accumulated_sample_num_list = self.make_dataset()
        self._logger.info('Found {} {} instances.'.format(self.accumulated_sample_num_list[-1], self.mode))

    def __len__(self):
        """ Return the total number of samples
        """
        return self.accumulated_sample_num_list[-1]

    def __getitem__(self, index):
        """ Generates one sample of data
        """
        if index in self.cache:
            pc, pose = self.cache[index]
        else:
            pc, pose = self.get_data(index)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (pc, pose)

        data = {'points': pc,  # 'pose': pose,
                'idx': np.array(index, dtype=np.int32)}

        if self.transform:
            data = self.transform(data)

        return data

    def get_data(self, index):
        """ Read data from npy/npz file and pre-process the data
        Return pc_np: 8 channel [x, y, z, nx, ny, nz, curvature, reflectance \in[0, 0.99] mean 0.27]
               pose: 4x4
        """
        # determine the sequence
        for i, accumulated_sample_num in enumerate(self.accumulated_sample_num_list):
            if index < accumulated_sample_num:
                break

        folder = self.folder_list[i]
        seq = self.seq_list[i]

        if i == 0:
            index_in_seq = index
        else:
            index_in_seq = index - self.accumulated_sample_num_list[i-1]
        pc_np_file = os.path.join(folder, '%06d.npy' % index_in_seq)
        pose_np_file = os.path.join(self.root_path, 'poses', '%02d'%seq, '%06d.npz'%index_in_seq)

        pc_np = np.load(pc_np_file)
        pose_np = np.load(pose_np_file)['pose']

        pc_np = process_point_cloud(pc_np, r_min=3.0, r_max=60.0, z_min=-3.0, z_max=10.0, grid_size=0.0)
        return pc_np[:, :self.feat_len], pose_np

    def get_seq_pose_by_index(self, index):
        # determine the sequence
        for i, accumulated_sample_num in enumerate(self.accumulated_sample_num_list):
            if index < accumulated_sample_num:
                break
        folder = self.folder_list[i]
        seq = self.seq_list[i]

        if i == 0:
            index_in_seq = index
        else:
            index_in_seq = index - self.accumulated_sample_num_list[i - 1]
        pose_np_file = os.path.join(self.root_path, 'poses', '%02d' % seq, '%06d.npz' % index_in_seq)
        pose_np = np.load(pose_np_file)['pose']

        return i, seq, index_in_seq, pose_np

    def make_dataset(self):
        if self.mode == 'train':
            seq_list = list(range(9))
        else:
            seq_list = [9, 10]

        # filter or not
        np_folder = 'np_0.20_20480_r90_sn'

        accumulated_sample_num = 0
        sample_num_list = []
        accumulated_sample_num_list = []
        folder_list = []
        for seq in seq_list:
            folder = os.path.join(self.root_path, np_folder, '%02d'%seq)
            folder_list.append(folder)

            sample_num = round(len(os.listdir(folder)))
            accumulated_sample_num += sample_num
            sample_num_list.append(sample_num)
            accumulated_sample_num_list.append(round(accumulated_sample_num))

        return seq_list, folder_list, sample_num_list, accumulated_sample_num_list


class ModelNet(Dataset):
    def __init__(self, dataset_path, subset='train', categories=None, transform=None):
        """ModelNet40 dataset from PointNet.
        Automatically downloads the dataset if not available

        Args:
            dataset_path (str): Folder containing processed dataset
            subset (str): Dataset subset, either 'train' or 'test'
            categories (list): Categories to use
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(ModelNet, self).__init__()

        self._logger = logging.getLogger(self.__class__.__name__)
        self._root = dataset_path
        self.transform = transform

        metadata_fpath = os.path.join(self._root, '{}_files.txt'.format(subset))
        self._logger.info('Loading data from {} for {}'.format(metadata_fpath, subset))

        if not os.path.exists(os.path.join(dataset_path)):
            self._logger.info('Invalid path: {}'.format(dataset_path))

        with open(os.path.join(dataset_path, 'shape_names.txt')) as fid:
            self._classes = [l.strip() for l in fid]
            self._category2idx = {e[1]: e[0] for e in enumerate(self._classes)}
            self._idx2category = self._classes

        with open(os.path.join(dataset_path, '{}_files.txt'.format(subset))) as fid:
            h5_filelist = [line.strip() for line in fid]
            h5_filelist = [x.replace('data/modelnet40_ply_hdf5_2048/', '') for x in h5_filelist]
            h5_filelist = [os.path.join(self._root, f) for f in h5_filelist]

        if categories is not None:
            categories_idx = [self._category2idx[c] for c in categories]
            self._logger.info('Categories used: {}.'.format(categories_idx))
            self._classes = categories
        else:
            categories_idx = None
            self._logger.info('Using all categories.')

        self._data, self._labels = self._read_h5_files(h5_filelist, categories_idx)
        self._logger.info('Loaded {} {} instances.'.format(self._data.shape[0], subset))

    def __getitem__(self, item):
        sample = {'points': self._data[item, :, :],
                   'label': self._labels[item],
                     'idx': np.array(item, dtype=np.int32)}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self._data.shape[0]

    @property
    def classes(self):
        return self._classes

    @staticmethod
    def _read_h5_files(fnames, categories):

        all_data = []
        all_labels = []

        for fname in fnames:
            f = h5py.File(fname, mode='r')
            data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1)
            labels = f['label'][:].flatten().astype(np.int64)

            if categories is not None:  # Filter out unwanted categories
                mask = np.isin(labels, categories).flatten()
                data = data[mask, ...]
                labels = labels[mask, ...]

            all_data.append(data)
            all_labels.append(labels)

        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return all_data, all_labels

    def to_category(self, i):
        return self._idx2category[i]


class SemanticKITTI(Dataset):
    def __init__(self, data_path, subset='train', return_ref=False):
        self.return_ref = return_ref
        self.filelist = "./dataloader/semantic-kitti.yaml"

        with open(self.filelist, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.subset = subset
        if subset == 'train':
            split = semkittiyaml['split']['train']
        elif subset == 'val':
            split = semkittiyaml['split']['valid']
        elif subset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        for i_folder in split:
            self.im_idx += self.absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))

        # load Semantic_KITTI class info
        self.label_name = dict()
        for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
            self.label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))

        # load labels
        if self.subset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3]+'label',
                                dtype=np.int32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)
        return data_tuple

    def get_unique_label(self):
        unique_label = np.asarray(sorted(list(self.label_name.keys())))[1:] - 1
        unique_label_str = [self.label_name[x] for x in unique_label + 1]
        return unique_label, unique_label_str

    def absoluteFilePaths(self, directory):
        for dirpath, _, filenames in os.walk(directory):
            for f in filenames:
                yield os.path.abspath(os.path.join(dirpath, f))



if __name__=='__main__':
    import torch
    from arguments import train_arguments

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # _device = torch.device('cuda:0')

    parser = train_arguments()
    args = parser.parse_args()

    train_set, val_set = get_train_datasets(args)
    dataset_loader = torch.utils.data.DataLoader(train_set,
                                                batch_size=2,
                                                collate_fn=train_set.collate_fn,
                                                shuffle=False,
                                                num_workers=1)
    for i, data in enumerate(dataset_loader):
        print(len(data))

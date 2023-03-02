import os
import json
import cv2
import numpy as np
from PIL import Image
from scipy import io
from skimage import morphology
import torch


class PASCALContext(torch.utils.data.Dataset):
    """
    credit: https://github.com/facebookresearch/astmt
    PASCAL-Context dataset, for multiple tasks
    Included tasks:
        1. Edge detection,
        2. Semantic Segmentation,
        3. Human Part Segmentation,
        4. Surface Normal prediction (distilled),
        5. Saliency (distilled)
    """

    HUMAN_PART = {1: {'hair': 1, 'head': 1, 'lear': 1, 'lebrow': 1, 'leye': 1, 'lfoot': 1,
                      'lhand': 1, 'llarm': 1, 'llleg': 1, 'luarm': 1, 'luleg': 1, 'mouth': 1,
                      'neck': 1, 'nose': 1, 'rear': 1, 'rebrow': 1, 'reye': 1, 'rfoot': 1,
                      'rhand': 1, 'rlarm': 1, 'rlleg': 1, 'ruarm': 1, 'ruleg': 1, 'torso': 1},
                  4: {'hair': 1, 'head': 1, 'lear': 1, 'lebrow': 1, 'leye': 1, 'lfoot': 4,
                      'lhand': 3, 'llarm': 3, 'llleg': 4, 'luarm': 3, 'luleg': 4, 'mouth': 1,
                      'neck': 2, 'nose': 1, 'rear': 1, 'rebrow': 1, 'reye': 1, 'rfoot': 4,
                      'rhand': 3, 'rlarm': 3, 'rlleg': 4, 'ruarm': 3, 'ruleg': 4, 'torso': 2},
                  6: {'hair': 1, 'head': 1, 'lear': 1, 'lebrow': 1, 'leye': 1, 'lfoot': 6,
                      'lhand': 4, 'llarm': 4, 'llleg': 6, 'luarm': 3, 'luleg': 5, 'mouth': 1,
                      'neck': 2, 'nose': 1, 'rear': 1, 'rebrow': 1, 'reye': 1, 'rfoot': 6,
                      'rhand': 4, 'rlarm': 4, 'rlleg': 6, 'ruarm': 3, 'ruleg': 5, 'torso': 2},
                  14: {'hair': 1, 'head': 1, 'lear': 1, 'lebrow': 1, 'leye': 1, 'lfoot': 14,
                       'lhand': 8, 'llarm': 7, 'llleg': 13, 'luarm': 6, 'luleg': 12, 'mouth': 1,
                       'neck': 2, 'nose': 1, 'rear': 1, 'rebrow': 1, 'reye': 1, 'rfoot': 11,
                       'rhand': 5, 'rlarm': 4, 'rlleg': 10, 'ruarm': 3, 'ruleg': 9, 'torso': 2}
                  }

    VOC_CATEGORY_NAMES = ['background',
                          'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                          'bus', 'car', 'cat', 'chair', 'cow',
                          'diningtable', 'dog', 'horse', 'motorbike', 'person',
                          'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    CONTEXT_CATEGORY_LABELS = [0,
                               2, 23, 25, 31, 34,
                               45, 59, 65, 72, 98,
                               397, 113, 207, 258, 284,
                               308, 347, 368, 416, 427]

    semseg_num_classes = 21
    edge_pos_weight = 0.95
    edge_tolerance = 0.0075

    image_dims = (3, 512, 512)

    def __init__(self,
                 data_dir,
                 split='train',
                 transforms=None,
                 area_thres=0,
                 retname=True,
                 tasks=('semseg',),
                 num_human_parts=6):
        self.root = os.path.join(data_dir, 'PASCALContext')

        centroids_path = os.path.join(data_dir, 'PASCALContext', 'centroids.npy')
        if os.path.exists(centroids_path):
            self.normals_centroids = torch.from_numpy(np.load(centroids_path).astype(np.float32))
        else:
            self.normals_centroids = None

        image_dir = os.path.join(self.root, 'JPEGImages')

        assert isinstance(split, str)
        self.split = [split]

        self.transforms = transforms
        self.area_thres = area_thres
        self.retname = retname

        # Edge Detection
        self.do_edge = ('edge' in tasks)
        self.edges = []
        edge_gt_dir = os.path.join(self.root, 'pascal-context', 'trainval')

        # Semantic Segmentation
        # self.do_semseg = ('semseg' in tasks)
        self.do_semseg = True
        self.semsegs = []

        # Human Part Segmentation
        self.do_human_parts = ('human_parts' in tasks)
        part_gt_dir = os.path.join(self.root, 'human_parts')
        self.parts = []
        self.human_parts_category = 15
        self.cat_part = json.load(open(os.path.join(os.path.dirname(__file__), 'db_info/pascal_part.json'), 'r'))
        self.cat_part["15"] = self.HUMAN_PART[num_human_parts]
        self.parts_file = os.path.join(self.root, 'ImageSets', 'Parts', ''.join(self.split) + '.txt')

        # Surface Normal Estimation
        self.do_normals = ('normals' in tasks)
        _normal_gt_dir = os.path.join(self.root, 'normals_distill')
        self.normals = []
        if self.do_normals:
            with open(os.path.join(os.path.dirname(__file__), 'db_info/nyu_classes.json')) as f:
                cls_nyu = json.load(f)
            with open(os.path.join(os.path.dirname(__file__), 'db_info/context_classes.json')) as f:
                cls_context = json.load(f)

            self.normals_valid_classes = []
            for cl_nyu in cls_nyu:
                if cl_nyu in cls_context and cl_nyu != 'unknown':
                    self.normals_valid_classes.append(cls_context[cl_nyu])

            # Custom additions due to incompatibilities
            self.normals_valid_classes.append(cls_context['tvmonitor'])

        # Saliency
        self.do_sal = ('sal' in tasks)
        _sal_gt_dir = os.path.join(self.root, 'sal_distill')
        self.sals = []

        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(self.root, 'ImageSets', 'Context')

        self.im_ids = []
        self.images = []

        for splt in self.split:
            with open(os.path.join(_splits_dir, splt + '.txt'), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                # Images
                _image = os.path.join(image_dir, line + ".jpg")
                assert os.path.isfile(_image), _image
                self.images.append(_image)
                self.im_ids.append(line.rstrip('\n'))

                # Edges
                _edge = os.path.join(edge_gt_dir, line + ".mat")
                assert os.path.isfile(_edge), _edge
                self.edges.append(_edge)

                # Semantic Segmentation
                _semseg = self._get_semseg_fname(line)
                assert os.path.isfile(_semseg), _semseg
                self.semsegs.append(_semseg)

                # Human Parts
                _human_part = os.path.join(part_gt_dir, line + ".mat")
                assert os.path.isfile(_human_part), _human_part
                self.parts.append(_human_part)

                _normal = os.path.join(_normal_gt_dir, line + ".png")
                assert os.path.isfile(_normal), _normal
                self.normals.append(_normal)

                _sal = os.path.join(_sal_gt_dir, line + ".png")
                assert os.path.isfile(_sal), _sal
                self.sals.append(_sal)

        if self.do_edge:
            assert len(self.images) == len(self.edges)
        if self.do_human_parts:
            assert len(self.images) == len(self.parts)
        if self.do_semseg:
            assert len(self.images) == len(self.semsegs)
        if self.do_normals:
            assert len(self.images) == len(self.normals)
        if self.do_sal:
            assert len(self.images) == len(self.sals)

        if not self._check_preprocess_parts():
            self._preprocess_parts()

        if self.do_human_parts:
            # Find images which have human parts
            self.has_human_parts = []
            for ii in range(len(self.im_ids)):
                if self.human_parts_category in self.part_obj_dict[self.im_ids[ii]]:
                    self.has_human_parts.append(1)
                else:
                    self.has_human_parts.append(0)

            # If the other tasks are disabled, select only the images that contain human parts,
            # to allow batching
            if not self.do_edge and not self.do_semseg and not self.do_sal and not self.do_normals:
                for i in range(len(self.parts) - 1, -1, -1):
                    if self.has_human_parts[i] == 0:
                        del self.im_ids[i]
                        del self.images[i]
                        del self.parts[i]
                        del self.has_human_parts[i]

    def __getitem__(self, index):
        sample = {}

        _img, lab_shape = self._load_img(index)
        sample['image'] = _img

        if self.do_edge:
            _edge = self._load_edge(index)
            assert _edge.shape[:2] == lab_shape
            sample['edge'] = _edge

        if self.do_human_parts:
            _human_parts, _ = self._load_human_parts(index, lab_shape)
            assert _human_parts.shape[:2] == lab_shape
            sample['human_parts'] = _human_parts

        if self.do_semseg:
            _semseg = self._load_semseg(index)
            assert _semseg.shape[:2] == lab_shape
            sample['semseg'] = _semseg

        if self.do_normals:
            _normals = self._load_normals_distilled(index)
            assert _normals.shape[:2] == lab_shape
            sample['normals'] = _normals

        if self.do_sal:
            _sal = self._load_sal_distilled(index)
            assert _sal.shape[:2] == lab_shape
            sample['sal'] = _sal

        if self.retname:
            sample['meta'] = {'image': str(self.im_ids[index]),
                              'im_size': (_img.shape[0], _img.shape[1])}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        lab_shape = _img.size[::-1]
        _img = np.array(_img, dtype=np.float32)
        return _img, lab_shape

    def _load_edge(self, index):
        # Read Target object
        _tmp = io.loadmat(self.edges[index])['LabelMap']
        _edge = cv2.Laplacian(_tmp, cv2.CV_64F)
        _edge = morphology.thin(np.abs(_edge) > 0).astype(np.float32)
        _edge = np.expand_dims(_edge, axis=2)
        return _edge

    def _load_human_parts(self, index, lab_shape):
        if self.has_human_parts[index]:

            # Read Target object
            _part_mat = io.loadmat(self.parts[index])['anno'][0][0][1][0]
            _inst_mask = _target = None

            for _obj in _part_mat:

                has_human = _obj[1][0][0] == self.human_parts_category
                has_parts = len(_obj[3]) != 0

                if has_human and has_parts:
                    _inter = _obj[2].astype(np.float32)
                    if _inst_mask is None:
                        _inst_mask = _inter
                        _target = np.zeros(_inst_mask.shape)
                    else:
                        _inst_mask = np.maximum(_inst_mask, _inter)

                    n_parts = len(_obj[3][0])
                    for part_i in range(n_parts):
                        cat_part = str(_obj[3][0][part_i][0][0])
                        mask_id = self.cat_part[str(self.human_parts_category)][cat_part]
                        mask = _obj[3][0][part_i][1].astype(bool)
                        _target[mask] = mask_id

            if _target is not None:
                _target, _inst_mask = _target.astype(np.float32), _inst_mask.astype(np.float32)
            else:
                _target, _inst_mask = np.zeros(lab_shape, dtype=np.float32), np.zeros(lab_shape, dtype=np.float32)
            return np.expand_dims(_target, axis=2), np.expand_dims(_inst_mask, axis=2)

        return np.expand_dims(np.zeros(lab_shape, dtype=np.float32), axis=2), np.expand_dims(np.zeros(lab_shape, dtype=np.float32), axis=2)

    def _load_semseg(self, index):
        _semseg = Image.open(self.semsegs[index])
        _semseg = np.expand_dims(np.array(_semseg, dtype=np.float32), axis=2)
        return _semseg

    def _load_normals_distilled(self, index):
        _tmp = Image.open(self.normals[index])
        _tmp = np.array(_tmp, dtype=np.float32)
        _tmp = 2.0 * _tmp / 255.0 - 1.0

        labels = io.loadmat(os.path.join(self.root, 'pascal-context', 'trainval', self.im_ids[index] + '.mat'))
        labels = labels['LabelMap']
        _normals = np.zeros(_tmp.shape, dtype=np.float)
        for x in np.unique(labels):
            if x in self.normals_valid_classes:
                _normals[labels == x, :] = _tmp[labels == x, :]
        return _normals

    def _load_sal_distilled(self, index):
        _sal = Image.open(self.sals[index])
        _sal = np.expand_dims(np.array(_sal, dtype=np.float32), axis=2) / 255.
        _sal = (_sal > 0.5).astype(np.float32)
        return _sal

    def _get_semseg_fname(self, fname):
        fname_voc = os.path.join(self.root, 'semseg', 'VOC12', fname + '.png')
        fname_context = os.path.join(
            self.root, 'semseg', 'pascal-context', fname + '.png')
        if os.path.isfile(fname_voc):
            seg = fname_voc
        elif os.path.isfile(fname_context):
            seg = fname_context
        else:
            seg = None
        return seg

    def _check_preprocess_parts(self):
        _obj_list_file = self.parts_file
        if not os.path.isfile(_obj_list_file):
            return False
        self.part_obj_dict = json.load(open(_obj_list_file, 'r'))
        return list(np.sort([str(x) for x in self.part_obj_dict.keys()])) == list(np.sort(self.im_ids))

    def _preprocess_parts(self):
        self.part_obj_dict = {}
        obj_counter = 0
        for ii in range(len(self.im_ids)):
            # Read object masks and get number of objects
            part_mat = io.loadmat(os.path.join(self.root, 'human_parts', '{}.mat'.format(self.im_ids[ii])))
            n_obj = len(part_mat['anno'][0][0][1][0])

            # Get the categories from these objects
            _cat_ids = []
            for jj in range(n_obj):
                obj_area = np.sum(part_mat['anno'][0][0][1][0][jj][2])
                if obj_area > self.area_thres:
                    _cat_ids.append(int(part_mat['anno'][0][0][1][0][jj][1]))
                else:
                    _cat_ids.append(-1)
                obj_counter += 1

            self.part_obj_dict[self.im_ids[ii]] = _cat_ids

        with open(self.parts_file, 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(self.im_ids[0], json.dumps(self.part_obj_dict[self.im_ids[0]])))
            for ii in range(1, len(self.im_ids)):
                outfile.write(',\n\t"{:s}": {:s}'.format(self.im_ids[ii], json.dumps(self.part_obj_dict[self.im_ids[ii]])))
            outfile.write('\n}\n')

    def __repr__(self):
        return self.__class__.__name__ + '()'

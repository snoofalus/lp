# Authors: A. Iscen, G. Tolias, Y. Avrithis, O. Chum. 2018.

#import torch.utils.data as data
from torchvision.datasets.vision import VisionDataset
from PIL import Image

import os
import os.path
import sys
import pdb

import numpy as np
import time

import faiss
from faiss import normalize_L2

from .diffusion import *
import scipy
import torch.nn.functional as F
import torch

import scipy.stats

import pickle
from typing import Any, Callable, cast, Dict, List, Optional, Tuple



def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)



def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        #self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

        imfile_name = '%s/images.pkl' % self.root
        if os.path.isfile(imfile_name):
            with open(imfile_name, 'rb') as f:
                self.images = pickle.load(f)
        else:
            self.images = None

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        
        path, target = self.samples[index]

        if (index not in self.labeled_idx):
            target = self.p_labels[index]

        weight = self.p_weights[index]

        if self.images is not None:
            sample = Image.fromarray(self.images[index,:,:,:])
        else:
            sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        
        c_weight = self.class_weights[target]

        return sample, target, weight, c_weight

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.npy')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

###
#Datafolder Images
###
#----------------------------------------------------------------------------------------------------

class DBSS(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(DBSS, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform = transform,
                                          target_transform = target_transform)
        self.imgs = self.samples
        self.pos_list = dict()
        self.pos_w = dict()
        self.labeled_idx = []
        self.unlabeled_idx = []
        self.all_labels = []
        self.pos_dist = dict()

        self.p_labels = []
        self.p_weights = np.ones((len(self.imgs),))
        self.class_weights = np.ones((len(self.classes),),dtype = np.float32)

        self.images_lists = [[] for i in range(len(self.classes))]

    def update_plabels(self, X, k = 50, max_iter = 20):

        print('Updating pseudo-labels...')
        alpha = 0.99
        labels = np.asarray(self.all_labels)
        labeled_idx = np.asarray(self.labeled_idx)
        unlabeled_idx = np.asarray(self.unlabeled_idx)

        
        # kNN search for the graph
        d = X.shape[1]
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = int(torch.cuda.device_count()) - 1
        index = faiss.GpuIndexFlatIP(res,d,flat_config)   # build the index

        normalize_L2(X)
        index.add(X) 
        N = X.shape[0]
        Nidx = index.ntotal

        c = time.time()
        D, I = index.search(X, k + 1)
        elapsed = time.time() - c
        print('kNN Search done in %d seconds' % elapsed)

        # Create the graph
        D = D[:,1:] ** 3
        I = I[:,1:]
        row_idx = np.arange(N)
        row_idx_rep = np.tile(row_idx,(k,1)).T
        W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
        W = W + W.T

        # Normalize the graph
        W = W - scipy.sparse.diags(W.diagonal())
        S = W.sum(axis = 1)
        S[S==0] = 1
        D = np.array(1./ np.sqrt(S))
        D = scipy.sparse.diags(D.reshape(-1))
        Wn = D * W * D

        # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
        Z = np.zeros((N,len(self.classes)))
        A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
        for i in range(len(self.classes)):
            cur_idx = labeled_idx[np.where(labels[labeled_idx] ==i)]
            y = np.zeros((N,))
            y[cur_idx] = 1.0 / cur_idx.shape[0]
            f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
            Z[:,i] = f

        # Handle numberical errors
        Z[Z < 0] = 0 

        # Compute the weight for each instance based on the entropy (eq 11 from the paper)
        probs_l1 = F.normalize(torch.tensor(Z),1).numpy()
        probs_l1[probs_l1 <0] = 0
        entropy = scipy.stats.entropy(probs_l1.T)
        weights = 1 - entropy / np.log(len(self.classes))
        weights = weights / np.max(weights)
        p_labels = np.argmax(probs_l1,1)

        # Compute the accuracy of pseudolabels for statistical purposes
        correct_idx = (p_labels == labels)
        acc = correct_idx.mean()

        p_labels[labeled_idx] = labels[labeled_idx]
        weights[labeled_idx] = 1.0

        self.p_weights = weights.tolist()
        self.p_labels = p_labels

        # Compute the weight for each class
        for i in range(len(self.classes)):
            cur_idx = np.where(np.asarray(self.p_labels) == i)[0]
            self.class_weights[i] = (float(labels.shape[0]) / len(self.classes)) / cur_idx.size

        return acc

###
#Datafolder for images, accepts npy
###
#----------------------------------------------------------------------------------------------------

class DBSS(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(DBSS, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform = transform,
                                          target_transform = target_transform)
        self.imgs = self.samples
        self.pos_list = dict()
        self.pos_w = dict()
        self.labeled_idx = []
        self.unlabeled_idx = []
        self.all_labels = []
        self.pos_dist = dict()

        self.p_labels = []
        self.p_weights = np.ones((len(self.imgs),))
        self.class_weights = np.ones((len(self.classes),),dtype = np.float32)

        self.images_lists = [[] for i in range(len(self.classes))]

    def update_plabels(self, X, k = 50, max_iter = 20):

        print('Updating pseudo-labels...')
        alpha = 0.99
        labels = np.asarray(self.all_labels)
        labeled_idx = np.asarray(self.labeled_idx)
        unlabeled_idx = np.asarray(self.unlabeled_idx)

        
        # kNN search for the graph
        d = X.shape[1]
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = int(torch.cuda.device_count()) - 1
        index = faiss.GpuIndexFlatIP(res,d,flat_config)   # build the index

        normalize_L2(X)
        index.add(X) 
        N = X.shape[0]
        Nidx = index.ntotal

        c = time.time()
        D, I = index.search(X, k + 1)
        elapsed = time.time() - c
        print('kNN Search done in %d seconds' % elapsed)

        # Create the graph
        D = D[:,1:] ** 3
        I = I[:,1:]
        row_idx = np.arange(N)
        row_idx_rep = np.tile(row_idx,(k,1)).T
        W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
        W = W + W.T

        # Normalize the graph
        W = W - scipy.sparse.diags(W.diagonal())
        S = W.sum(axis = 1)
        S[S==0] = 1
        D = np.array(1./ np.sqrt(S))
        D = scipy.sparse.diags(D.reshape(-1))
        Wn = D * W * D

        # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
        Z = np.zeros((N,len(self.classes)))
        A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
        for i in range(len(self.classes)):
            cur_idx = labeled_idx[np.where(labels[labeled_idx] ==i)]
            y = np.zeros((N,))
            y[cur_idx] = 1.0 / cur_idx.shape[0]
            f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
            Z[:,i] = f

        # Handle numberical errors
        Z[Z < 0] = 0 

        # Compute the weight for each instance based on the entropy (eq 11 from the paper)
        probs_l1 = F.normalize(torch.tensor(Z),1).numpy()
        probs_l1[probs_l1 <0] = 0
        entropy = scipy.stats.entropy(probs_l1.T)
        weights = 1 - entropy / np.log(len(self.classes))
        weights = weights / np.max(weights)
        p_labels = np.argmax(probs_l1,1)

        # Compute the accuracy of pseudolabels for statistical purposes
        correct_idx = (p_labels == labels)
        acc = correct_idx.mean()

        p_labels[labeled_idx] = labels[labeled_idx]
        weights[labeled_idx] = 1.0

        self.p_weights = weights.tolist()
        self.p_labels = p_labels

        # Compute the weight for each class
        for i in range(len(self.classes)):
            cur_idx = np.where(np.asarray(self.p_labels) == i)[0]
            self.class_weights[i] = (float(labels.shape[0]) / len(self.classes)) / cur_idx.size

        return acc
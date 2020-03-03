from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import codecs
import numpy as np

# Code referenced from torch source code to add Fashion-MNSIT dataset to dataloder
# Url: http://pytorch.org/docs/0.3.0/_modules/torchvision/datasets/mnist.html#FashionMNIST
class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, few_shot_class=None, test_emnist=True, max_test_sample=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.few_shot_class = few_shot_class
        self.test_emnist = test_emnist
        self.max_test_sample = max_test_sample
        print("MNIST: will we ignore download")
        if download:
            print("MNIST: trying to download")
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        print("download: trying to download")
        if self._check_exists():
            print("download: already exists so exiting")
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')
        train_label, train_non_few_shot_ids, train_few_shot_ids =  read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'), self.few_shot_class)
        train_img = read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte'), non_few_shot_ids=train_non_few_shot_ids)
        
        training_set = (
            train_img,
            train_label
        )
        
        test_label, test_non_few_shot_ids, test_few_shot_ids=  read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'), self.few_shot_class)
        test_img = read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte'), few_shot_ids=test_few_shot_ids)
        
            
        if self.test_emnist:
           print("Download: Entering Emnist test")
           from emnist import extract_test_samples
           images, labels = extract_test_samples('letters')
           print(images.shape)
           print(labels.shape) 
           #randomly grab a letter 
           import random
           rand_letter_idx = random.randint(0,25)
           #idx for selected letter clas
           test_sample_ids = np.where(labels < 10)[0]
           np.random.seed(10)
           np.random.shuffle(test_sample_ids)
           
           print('test_sample_ids_len' , len(test_sample_ids))
           #grab labels and images from that class
           labels = labels[test_sample_ids]
           images = images[test_sample_ids]           
           print("After selecting one class")
           print(images.shape)
           print(labels.shape)
           #assert(self.few_shot_class not in labels)
           if self.max_test_sample:
             test_set = {
                torch.ByteTensor(list(images[:self.max_test_sample])).view(-1,28,28),
                torch.LongTensor(list(labels[:self.max_test_sample]))
             }
           else:
             test_set = {
                 torch.ByteTensor(list(images)).view(-1,28,28),
                 torch.LongTensor(list(labels))
              }  
        else:
        # test_label, test_non_few_shot_ids, test_few_shot_ids=  read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'), self.few_shot_class)
        # test_img = read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte'), few_shot_ids=test_few_shot_ids)
           if (self.max_test_sample):
               print('testing max test sample')
               test_set = (
                    test_img[:self.max_test_sample],
                    test_label[:self.max_test_sample]
               )
      
           else:
               test_set = (
                    test_img,
                    test_label
               )  
        print('confirming test size')
        #print(len(test_set[0]), len(test_set[1]))         
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')


class FashionMNIST(MNIST):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.
           images = np.array(images)
           images = np.array(images)
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def parse_byte(b):
    if isinstance(b, str):
        return ord(b)
    return b


def read_label_file(path, few_shot_class):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        labels = [parse_byte(b) for b in data[8:]]
        assert len(labels) == length
        if few_shot_class:
         labels = np.array(labels)
         non_few_shot_ids = np.where(labels != few_shot_class)
         few_shot_ids = np.where(labels == few_shot_class)
         labels = labels[non_few_shot_ids]
         assert(few_shot_class not in labels)
         return torch.LongTensor(list(labels)), non_few_shot_ids , few_shot_ids 

        else:  
         #error where longtensor not working 
         return torch.LongTensor(labels)

def read_image_file(path, non_few_shot_ids=None, few_shot_ids=None):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        idx = 16
        for l in range(length):
            img = []
            images.append(img)
            for r in range(num_rows):
                row = []
                img.append(row)
                for c in range(num_cols):
                    row.append(parse_byte(data[idx]))
                    idx += 1
        if non_few_shot_ids:
         print("Preparing training dataset")
         images = np.array(images)
         print(len(images))
         images = images[non_few_shot_ids] 
         print(len(images))
         assert(len(images) == len(non_few_shot_ids[0])) 
         return torch.ByteTensor(list(images)).view(-1, 28, 28)
        elif few_shot_ids:
         print("Preparring testing dataset")
         print("read_image_file: getting testing dataset")
         images = np.array(images)
         print(len(images))
         images = images[few_shot_ids]
         print(len(images))
         assert(len(images) == len(few_shot_ids[0])) 
         return torch.ByteTensor(list(images)).view(-1, 28, 28) 
        else:
         assert len(images) == length
         return torch.ByteTensor(images).view(-1,28,28)

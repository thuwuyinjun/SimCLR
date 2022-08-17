from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection


from torch.utils.data import Subset, Dataset, DataLoader
from PIL import Image
import numpy as np
import torch

class dataset_wrapper(Dataset):
    def __init__(self, data_tensor, label_tensor, transform, three_imgs = False, two_imgs = False):

        # super(new_mnist_dataset, self).__init__(*args, **kwargs)
        self.data = data_tensor
        self.targets = label_tensor
        self.transform = transform
        self.three_imgs = three_imgs
        self.two_imgs = two_imgs

    
    def clone_dataset(self):
        if not type(self.data) is np.ndarray:
            return dataset_wrapper(self.data.clone(), self.targets.clone(), self.transform, self.three_imgs, self.two_imgs) 
        else:
            return dataset_wrapper(np.copy(self.data), np.copy(self.targets), self.transform, self.three_imgs, self.two_imgs) 

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # if len(self.targets) < 100 and index == 0:
        #     print(self.targets)
        if not type(img) is np.ndarray:
            if type(img) is torch.Tensor:
                img = Image.fromarray(img.numpy(), mode="L")
            else:
                img = Image.fromarray(img.np(), mode="L")
        else:
            img = Image.fromarray(img)
        if self.transform is not None:
            try:
                img1 = self.transform(img)
            except:
                img1, target = self.transform(img, target)

            # if self.two_imgs:
            #     img2 = self.transform(img)
            #     return (img1, img2), target, index


            # if self.three_imgs:
            #     img2 = self.transform(img)
            #     img3 = self.transform(img)
            #     return (img1, img2, img3), target, index

        return (index, img1, target)
        # image, target = super(new_mnist_dataset, self).__getitem__(index)

        # return (index, image,target)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_subset_dataset(dataset, sample_ids, labels = None):
        subset_data = dataset.data[sample_ids]
        if labels is None:
            subset_labels = dataset.targets[sample_ids]
        else:
            subset_labels = labels[sample_ids]
        transform = dataset.transform


        three_imgs = dataset.three_imgs
        two_imgs = dataset.two_imgs

        if not type(dataset.data) is np.ndarray:
            subset_data = subset_data.clone()
            subset_labels = subset_labels.clone()
            if len(sample_ids) <= 1:
                subset_data = subset_data.unsqueeze(0)
                subset_labels = subset_labels.unsqueeze(0)
            
        else:
            subset_data = np.copy(subset_data)
            subset_labels = np.copy(subset_labels)
            if len(sample_ids) <= 1:
                subset_data = np.expand_dims(subset_data, 0)
                subset_labels = np.expand_dims(subset_labels, 0)

        return dataset_wrapper(subset_data, subset_labels, transform, three_imgs, two_imgs)
    
    @staticmethod
    def set_values_for_subset(dataset, sample_ids, sub_dataset):
        subset_data = sub_dataset.data
        subset_labels = sub_dataset.targets

        dataset.data[sample_ids] = subset_data
        dataset.targets[sample_ids] = subset_labels

    @staticmethod
    def subsampling_dataset_by_class(dataset, num_per_class=45):
        if type(dataset.data) is np.ndarray:
            label_set = np.unique(dataset.targets)
        else:
            label_set = torch.unique(dataset.targets)

        full_sel_sample_ids = []
        for label in label_set:
            if type(dataset.data) is np.ndarray:
                sample_ids_with_curr_labels = np.nonzero((dataset.targets == label))[0].reshape(-1)
                sample_ids_with_curr_labels = torch.from_numpy(sample_ids_with_curr_labels)
            else:
                sample_ids_with_curr_labels = torch.nonzero((dataset.targets == label)).reshape(-1)

            random_sample_ids_with_curr_labels = torch.randperm(len(sample_ids_with_curr_labels))

            selected_sample_ids_with_curr_labels = random_sample_ids_with_curr_labels[0:num_per_class]

            full_sel_sample_ids.append(selected_sample_ids_with_curr_labels)

        full_sel_sample_ids_tensor = torch.cat(full_sel_sample_ids)
        
        if type(dataset.data) is np.ndarray:
            return dataset.get_subset_dataset(dataset, full_sel_sample_ids_tensor.numpy())
        else:
            return dataset.get_subset_dataset(dataset, full_sel_sample_ids_tensor)




    @staticmethod
    def concat_validset(dataset1, dataset2):
        valid_data_mat = dataset1.data
        valid_labels = dataset1.targets
        if type(valid_data_mat) is np.ndarray:
            if len(dataset2.data.shape) < len(valid_data_mat.shape):
                dataset2.data = np.expand_dims(dataset2.data, 0)
                dataset2.targets = np.expand_dims(dataset2.targets,0)
            valid_data_mat = np.concatenate((valid_data_mat, dataset2.data), axis = 0)
            valid_labels = np.concatenate((valid_labels, dataset2.targets), axis = 0)
            
        else:
            if len(dataset2.data.shape) < len(valid_data_mat.shape):
                dataset2.data = dataset2.data.unsqueeze(0)
                dataset2.targets = dataset2.targets.unsqueeze(0)
            valid_data_mat = torch.cat([valid_data_mat, dataset2.data], dim = 0)
            valid_labels = torch.cat([valid_labels, dataset2.targets], dim = 0)
        valid_set = dataset_wrapper(valid_data_mat, valid_labels, dataset1.transform)
        return valid_set

    @staticmethod
    def to_cuda(data, targets):
        return data.cuda(), targets.cuda()


def subset_cifar(train_dataset, val_ratio = 0.05):
    shuffled_train_idx = torch.randperm(len(train_dataset))
    valid_count = int(len(train_dataset)*val_ratio)
    valid_idx = shuffled_train_idx[0:valid_count]
    train_idx = shuffled_train_idx[valid_count:]
    validset = train_dataset.get_subset_dataset(train_dataset, valid_idx)
    train_dataset = train_dataset.get_subset_dataset(train_dataset, train_idx)
    return train_dataset, validset



class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    @staticmethod
    def get_test_transform(size):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        # color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True)}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return_dataset=dataset_fn()
            if name == 'cifar10':
                labels = return_dataset.targets
                data = np.copy(return_dataset.data)
            elif name == 'stl10':
                labels = return_dataset.labels
                data = np.transpose(return_dataset.data, (0,2,3,1))

                # data = np.copy(return_dataset.data)


            return_dataset = dataset_wrapper(data, np.copy(labels), return_dataset.transforms)
            return return_dataset

    def get_eval_dataset(self, name, train=True):

        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=train,
                                                              transform=transforms.ToTensor(),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split=True,
                                                          transform=transforms.ToTensor(),
                                                          download=True)}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return_dataset=dataset_fn()
            if name == 'cifar10':
                labels = return_dataset.targets
                data = np.copy(return_dataset.data)
            elif name == 'stl10':
                labels = return_dataset.labels
                data = np.transpose(return_dataset.data, (0,2,3,1))

                # data = np.copy(return_dataset.data)


            return_dataset = dataset_wrapper(data, np.copy(labels), return_dataset.transforms)
            return return_dataset

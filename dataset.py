import random
import copy

import numpy as np
import torch
import torchio as tio
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader

from ct_augmentations import get_transform


class CTDataset3D(Dataset):
    """
    Dataset class for handling 3D CT scan images.

    Attributes:
        root (str): The directory where the dataset .npz file is located.
        augment (bool): Flag to indicate if data augmentation is enabled.
        transform (optional): Augmentation transform to be applied to the data.
        dataset_name (str): Name of the dataset.
        get_img_names (bool): Flag to return image names along with the data.
        imgs (numpy.ndarray): The array of input images.
        labels (numpy.ndarray): The array of labels (ground truth).
        pat_ids (numpy.ndarray): The array of patient IDs.
        img_names (numpy.ndarray): The array of image file names.

    Methods:
        extract_files():
            Loads and extracts images, labels, and metadata from the dataset stored in a .npz file.

        sample(ids):
            Subsets the dataset to a specific set of samples identified by the provided indices.

        min_max_scaler(img):
            Static method. Normalizes the image data by applying min-max scaling.

        monai_get_item(img):
            Prepares an image by applying transformations using the MONAI library.

        torch_io_get_item(img):
            Prepares an image by applying transformations using the TorchIO library.
        """

    def __init__(self,
                 root: str,
                 augm_transform=None,
                 dataset_name: str = "",
                 get_img_names: bool = False
                 ):
        """
        Parameters
        :param root: Root path of the .npz file.
        :param augm_transform: Augmentation transformations to apply.
        randomly alternate between TorchIO and MONAI. Default to 'both'.
        :param dataset_name: Name of the input dataset.
        :param get_img_names: Whether to return also image names in __get_item__ method (default False).
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.root = root
        self.get_img_names = get_img_names
        self.imgs, self.labels, self.pat_ids, self.img_names = self.extract_files()
        self.augment = augm_transform is not None  # whether to augment data or not
        self.transform = augm_transform

    def extract_files(self):
        """ Extract images, labels and redcap from .npz file previously created """
        print(f"loading numpy samples {self.dataset_name} [This may take some seconds]")
        npz_file = np.load(self.root)  # Loading npz files where signals and ExamIDs are stored
        imgs = npz_file["imgs"]
        labels = npz_file["labels"]
        pat_ids = npz_file["pat_ids"]
        img_names = npz_file["img_names"]
        print(f"{self.dataset_name} samples are loaded!")
        return imgs, labels, pat_ids, img_names

    def sample(self, ids):
        """
        Extract only the required ids. The rest of the dataset is dropped.
        :param ids: indices of the samples to select.
        :return:
        """
        self.labels = self.labels[ids]
        self.imgs = self.imgs[ids]
        self.pat_ids = self.pat_ids[ids]
        self.img_names = self.img_names[ids]

    @staticmethod
    def min_max_scaler(img):
        return (img - img.min()) / (img.max() - img.min())

    def __len__(self):
        return len(self.imgs)

    def monai_get_item(self, img):
        """
        Apply Monai augmentation to a given image.
        :param img: image to augment
        :return: augmented image
        """
        if self.augment:
            input_items = [{"img": img, "label": -1}]
            if not isinstance(self.transform, dict):
                img = self.transform(input_items)[0]['img']
            else:
                img = self.transform['monai'](input_items)[0]['img']
            # img = self.min_max_scaler(img)
        img = torch.Tensor(img)
        last_id = len(img.shape) - 1
        img = img.swapaxes(last_id - 1, last_id).swapaxes(last_id - 2, last_id - 1)
        img = img.unsqueeze(0)  # add channel dimension if 3D
        return img

    def torch_io_get_item(self, img):
        """
        Apply TorchIO augmentation to a given image.
        :param img: image to augment
        :return: augmented image
        """
        img = torch.Tensor(img).to(torch.float32)
        if self.augment:  # apply augmentation if required
            if not isinstance(self.transform, dict):
                img = self.transform(img.unsqueeze(0)).squeeze(0)
            else:
                img = self.transform['torch_io'](img.unsqueeze(0)).squeeze(0)
            img = self.min_max_scaler(img)
        last_id = len(img.shape) - 1
        img = img.swapaxes(last_id - 1, last_id).swapaxes(last_id - 2, last_id - 1)
        img = img.unsqueeze(0)  # add channel dimension if 3D
        return img

    def __getitem__(self, index):
        """Retrieves a specific sample (image, label) from the dataset, applying augmentation if required."""
        label = self.labels[index]
        img = self.imgs[index]
        # apply randomly torch_io or monai augmentation
        if random.random() < 0.5:
            img = self.monai_get_item(img)
        else:
            img = self.torch_io_get_item(img)
        if self.get_img_names:
            img_name = self.img_names[index]
            return img, label, img_name
        return img, label


def load_train(root_train: str, augm_list: list, batch_size: int, augm_params: dict,
               num_workers: int, val_fraction: float, augm_val: bool = False):
    """
    Creates training and validation datasets from a .npz file of CT images. It splits data into
    training and validation sets stratifying on labels and assuring that all the images from the same patient are put in
    the same set.

    :param root_train:  Path to the .npz file of the training dataset.
    :param augm_list: List of augmentation types to apply (used with TorchIO).
    :param batch_size: Number of samples per batch for the DataLoaders.
    :param augm_params: Parameters for the augmentation transformations.
    :param num_workers: Number of subprocesses for data loading.
    :param val_fraction: Fraction of the dataset to use for validation.
    :param augm_val: Whether to apply augmentations to the validation dataset (default False).
    :return: A tuple containing training and validation dataloaders.
    """
    # Define training and strain_test datasets
    augm_transforms = {"torch_io": torch_io_augmentation(augm_list, augm_params),
                       "monai": get_transform(augm_params["p"])[0]}

    train_dataset = CTDataset3D(root=root_train,
                                augm_transform=augm_transforms,
                                dataset_name="train")

    n_splits = int(round(1 / val_fraction, 0))
    sgkf = StratifiedGroupKFold(n_splits=n_splits)  # Grouping by patient id and stratifying based on labels
    splitter = sgkf.split(np.arange(len(train_dataset.labels)),
                          y=train_dataset.labels,
                          groups=train_dataset.pat_ids)
    train_ids, val_ids = next(splitter)  # Choosing the first split as validation!
    val_dataset = copy.deepcopy(train_dataset)
    train_dataset.sample(train_ids)
    val_dataset.sample(val_ids)
    val_dataset.augment = augm_val
    assert ~np.isin(train_dataset.pat_ids, val_dataset.pat_ids).any(), "Patients repeated in train and val sets"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, val_loader


def load_test(root: str, batch_size: int = 2):
    """
    Loads the test dataset and creates a DataLoader for inference.
    :param root: Path to the test dataset (in .npz format).
    :param batch_size: Dataloader batch size
    :return: DataLoader for the test dataset.
    """
    test_set = CTDataset3D(root=root, dataset_name=root)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return test_loader


def torch_io_augmentation(augm_list: list, augm_params: dict):
    """
    Creates a set of TorchIO augmentations based on the provided augmentation list and parameters.
    :param augm_list: List of augmentation names to apply (choose among "Gamma", "Noise", "Blur" and "Biasfield").
    :param augm_params: Dictionary containing parameters for each augmentation.
    :return: (tio.transforms.Transform) A TorchIO `OneOf` transform combining the specified augmentations,
        or `None` if no augmentations are provided.
    """
    augment_dict = {"Gamma": tio.RandomGamma(log_gamma=augm_params["log_gamma"]),
                    "Noise": tio.RandomNoise(mean=augm_params["noise_mean"], std=augm_params["noise_std"]),
                    "Blur": tio.RandomBlur(std=augm_params["blur_std"]),
                    "Biasfield": tio.RandomBiasField(coefficients=augm_params["biasfield"])
                    }
    augmentation = [augment_dict[augm] for augm in augm_list]
    augm_transforms = None if len(augmentation) == 0 else tio.OneOf(augmentation, p=augm_params["p"])
    return augm_transforms

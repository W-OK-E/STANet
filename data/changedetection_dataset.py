from data.base_dataset import BaseDataset, get_transform, get_params
from data.image_folder import make_dataset
from PIL import Image
import os
import numpy as np


class ChangeDetectionDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    datafolder-tree
    dataroot:.
        ├─train
        │   ├─A
        │   ├─B
        │   ├─label
        ├─val
        │   ├─A
        │   ├─B
        │   ├─label
        ├─test
            ├─A
            ├─B
            ├─label
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        root_split = os.path.join(opt.dataroot, opt.split)
        folder_A = os.path.join(root_split, 'A')
        folder_B = os.path.join(root_split, 'B')
        folder_L = os.path.join(root_split, 'label')

        self.istest = (opt.phase == 'test')

        self.A_paths = sorted(make_dataset(folder_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(folder_B, opt.max_dataset_size))
        self.L_paths = sorted(make_dataset(folder_L, opt.max_dataset_size)) if not self.istest else []

        # 🔧 Ensure all lists are aligned
        if not self.istest:
            min_len = min(len(self.A_paths), len(self.B_paths), len(self.L_paths))
            if not (len(self.A_paths) == len(self.B_paths) == len(self.L_paths)):
                print(f"[WARNING] Mismatch in dataset sizes: A={len(self.A_paths)}, "
                      f"B={len(self.B_paths)}, L={len(self.L_paths)}. "
                      f"Trimming to {min_len}.")
            self.A_paths = self.A_paths[:min_len]
            self.B_paths = self.B_paths[:min_len]
            self.L_paths = self.L_paths[:min_len]
        else:
            min_len = min(len(self.A_paths), len(self.B_paths))
            if not (len(self.A_paths) == len(self.B_paths)):
                print(f"[WARNING] Mismatch in dataset sizes: A={len(self.A_paths)}, "
                      f"B={len(self.B_paths)}. Trimming to {min_len}.")
            self.A_paths = self.A_paths[:min_len]
            self.B_paths = self.B_paths[:min_len]

        print(f"Loaded {len(self.A_paths)} samples from {root_split}")

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        transform_params = get_params(self.opt, A_img.size, test=self.istest)
        transform = get_transform(self.opt, transform_params, test=self.istest)

        A = transform(A_img)
        B = transform(B_img)

        if self.istest:
            return {'A': A, 'A_paths': A_path,
                    'B': B, 'B_paths': B_path}

        L_path = self.L_paths[index]
        tmp = np.array(Image.open(L_path), dtype=np.uint32) / 255
        L_img = Image.fromarray(tmp)
        transform_L = get_transform(self.opt, transform_params,
                                    method=Image.NEAREST, normalize=False,
                                    test=self.istest)
        L = transform_L(L_img)

        return {'A': A, 'A_paths': A_path,
                'B': B, 'B_paths': B_path,
                'L': L, 'L_paths': L_path}

    def __len__(self):
        return len(self.A_paths)

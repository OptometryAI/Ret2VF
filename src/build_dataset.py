import ast
import re

import numpy as np
import pandas as pd
import torch
from dataframe_dealer import get_df_length, analyze_and_get_labeld_data, stratified_split_by_md
from torch.utils.data import DataLoader, Dataset


N_CATEGORIES_PER_POINT = 50


class VFWithOCTDataset(Dataset):
    def __init__(self, dataframe, transform=None, n_categories_per_point=50):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.n_categories_per_point = n_categories_per_point

    def __len__(self):
        return len(self.dataframe)

    def get_3d_oct(self, data_row):
        oct_npy_path = data_row['processed_oct_path']
        oct_3d = np.load(oct_npy_path)
        oct_3d = torch.tensor(oct_3d, dtype=torch.float32)
        oct_3d = oct_3d.unsqueeze(0)
        return oct_3d

    def normalize_3d_oct(self, oct_3d):
        mean = torch.mean(oct_3d)
        std = torch.std(oct_3d)
        if std == 0:
            return oct_3d - mean
        return (oct_3d - mean) / std

    def clean_vf_in_table_to_list(self, data_row):
        grid_vf = data_row['HVF_VisualFieldPlot']
        rows = ast.literal_eval(grid_vf)
        extracted_values = []
        for row in rows:
            matches = re.findall(r'(<0|-?\d+)', row)
            for val in matches:
                if val == '<0':
                    extracted_values.append(0)
                else:
                    extracted_values.append(int(val))
        return extracted_values

    def turn_vf_from_right_to_left(self, vf):
        segments = [4, 6, 8, 9, 9, 8, 6, 4]
        start_idx = 0
        new_vf = []
        for segment_length in segments:
            segment = vf[start_idx:start_idx + segment_length]
            reversed_segment = segment[::-1]
            new_vf.extend(reversed_segment)
            start_idx += segment_length
        vf = new_vf
        vf[19] = 0
        vf[28] = 0
        return np.array(vf)

    def encode_vf_as_onehot(self, vf):
        vf_encoded = None
        for value in vf:
            value = int(max(0, min(value, self.n_categories_per_point)))
            vf_levels = np.array([[1] * value + [0] * (self.n_categories_per_point - value)])
            if vf_encoded is None:
                vf_encoded = vf_levels
            else:
                vf_encoded = np.concatenate((vf_encoded, vf_levels), axis=0)
        return vf_encoded

    def get_vf_tensor(self, vf):
        return torch.tensor(vf, dtype=torch.float32)

    def encode_pm_label(self, label):
        label_mapping = {'C0': 0, 'C1': 0, 'C2': 1, 'C3': 1, 'C4': 1}
        return label_mapping.get(label, -1)

    def __getitem__(self, idx):
        data_row = self.dataframe.iloc[idx]

        oct_3d = self.get_3d_oct(data_row)
        oct_3d = self.normalize_3d_oct(oct_3d)

        vf = self.clean_vf_in_table_to_list(data_row)
        if data_row['Laterality'] == 'R':
            vf = self.turn_vf_from_right_to_left(vf)
        vf_encoded = self.encode_vf_as_onehot(vf)
        vf_tensor = self.get_vf_tensor(vf)
        vf_encoded_tensor = self.get_vf_tensor(vf_encoded)

        pm_label = self.encode_pm_label(data_row['Meta-PM'])
        pm_label_tensor = torch.tensor(pm_label, dtype=torch.long)

        md_value = torch.tensor(float(data_row['HVF_MD']), dtype=torch.float32)

        laterality = data_row['Laterality']
        age = data_row['Age']
        pid = data_row['PID']

        return {
            'oct_3D': oct_3d,
            'VF_tensor': vf_tensor,
            'VF_encoded_tensor': vf_encoded_tensor,
            'pm_label': pm_label_tensor,
            'MD': md_value,
            'data_row': [pid, laterality, age],
        }


def divide_dataset(cfg):
    data_cfg = cfg['data']

    df = pd.read_excel(data_cfg['data_root'])
    dataset_ratio = data_cfg.get('dataset_ratio', [0.8, 0.1, 0.1])
    seed = cfg.get('seed', 42)

    all_length = get_df_length(df)
    train_len, val_len, test_len = [int(r * all_length) for r in dataset_ratio]

    print('Starting dataset partition analysis...')
    with_label_df, no_label_df = analyze_and_get_labeld_data(df)

    train_with_label_len = len(with_label_df)
    train_no_label_len = train_len - train_with_label_len
    sub_dataset_ratio = [
        train_no_label_len / len(no_label_df),
        val_len / len(no_label_df),
        test_len / len(no_label_df),
    ]
    train_no_label_df, val_no_label_df, test_no_label_df = stratified_split_by_md(
        no_label_df,
        ratios=sub_dataset_ratio,
        seed=seed,
    )

    train_df = pd.concat([with_label_df, train_no_label_df]).sample(frac=1, random_state=seed)
    val_df = val_no_label_df
    test_df = test_no_label_df
    return train_df, val_df, test_df


def make_dataloaders(cfg):
    data_cfg = cfg['data']
    train_df, val_df, test_df = divide_dataset(cfg)

    train_dataset = VFWithOCTDataset(
        train_df,
        transform=None,
        n_categories_per_point=data_cfg.get('n_categories_per_point', N_CATEGORIES_PER_POINT),
    )
    val_dataset = VFWithOCTDataset(
        val_df,
        transform=None,
        n_categories_per_point=data_cfg.get('n_categories_per_point', N_CATEGORIES_PER_POINT),
    )
    test_dataset = VFWithOCTDataset(
        test_df,
        transform=None,
        n_categories_per_point=data_cfg.get('n_categories_per_point', N_CATEGORIES_PER_POINT),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=True,
        num_workers=data_cfg.get('num_workers', 0),
        pin_memory=data_cfg.get('pin_memory', False),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 0),
        pin_memory=data_cfg.get('pin_memory', False),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 0),
        pin_memory=data_cfg.get('pin_memory', False),
    )

    return train_loader, val_loader, test_loader

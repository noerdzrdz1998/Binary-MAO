import os
import numpy as np
import pandas as pd
from typing import List, Tuple

def process_files(folder_path: str, seed: int = 42) -> Tuple[List[pd.DataFrame], List[str]]:
    """
    Processes data files in the specified folder, converts the data to numerical format, 
    and maps class labels to binary integer values.

    Parameters:
    -----------
    folder_path : str
        Path to the folder containing data files to be processed.
    seed : int, optional, default=42
        Random seed for reproducibility of the label assignment process.

    Returns:
    --------
    Tuple[List[pd.DataFrame], List[str]]:
        - A list of processed datasets (pandas DataFrames), where each DataFrame contains
          numerical feature columns and a binary `class` column.
        - A list of dataset names extracted from the filenames (without extensions).

    Raises:
    -------
    FileNotFoundError
        If the specified folder does not exist.
    ValueError
        If a file in the folder does not contain exactly two unique class labels.

    Notes:
    ------
    - Supports `.dat`, `.csv`, `.xlsx` files, assuming comma-separated values.
    - Missing values in the dataset are removed.
    - The last column is assumed to be the class label column.
    - If class labels are not `0` and `1` (in any format), they are randomly mapped to binary integer values.

    Example:
    --------
    >>> datasets, names = process_files("/path/to/data")
    >>> print(names)
    ['dataset1', 'dataset2']
    >>> print(datasets[0].head())
         0     1     2  class
    0  1.5  2.4  3.1      0
    1  3.2  4.1  5.0      1
    """
    np.random.seed(seed)

    datasets: List[pd.DataFrame] = []
    datasets_names: List[str] = []

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")

    supported_extensions = ('.dat', '.csv', '.xlsx')

    for file_name in os.listdir(folder_path):
        if not file_name.endswith(supported_extensions):
            continue

        file_path = os.path.join(folder_path, file_name)

        if file_name.endswith('.dat') or file_name.endswith('.csv'):
            df = pd.read_csv(file_path, header=None, sep=',')
        elif file_name.endswith('.xlsx'):
            df = pd.read_excel(file_path, header=None)

        df = df.dropna()

        class_column = df.iloc[:, -1]

        unique_classes = class_column.value_counts().index
        if len(unique_classes) != 2:
            raise ValueError(f"File '{file_name}' does not have exactly two unique class labels.")

        # Check if classes are binary (0 and 1 in any format: int, float, or str)
        try:
            if set(map(lambda x: int(float(x)) if str(x).replace('.', '', 1).isdigit() else x, unique_classes)) == {0, 1}:
                class_mapping = {uc: int(float(uc)) for uc in unique_classes}
            else:
                # Randomly assign binary labels (0 and 1) to other classes
                assigned_labels = np.random.permutation([0, 1])
                class_mapping = {unique_classes[0]: assigned_labels[0], 
                                 unique_classes[1]: assigned_labels[1]}
        except ValueError:
            # Handle cases where conversion to float/int fails (e.g., non-numeric labels like "cat", "dog")
            assigned_labels = np.random.permutation([0, 1])
            class_mapping = {unique_classes[0]: assigned_labels[0], 
                             unique_classes[1]: assigned_labels[1]}

        class_column = class_column.replace(class_mapping).astype(int)

        features = df.iloc[:, :-1].astype(float)
        features['class'] = class_column

        datasets.append(features)
        dataset_name = file_name.split('.')[0]
        datasets_names.append(dataset_name)

    return datasets, datasets_names

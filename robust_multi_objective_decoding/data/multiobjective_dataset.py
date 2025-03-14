# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 16:00:00 2024

@author: Abdelhamid

The file is an adaptation of PKUSafeRLHFDataset for the multi-objective setting.
"""

import pathlib
import random
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import datasets
from datasets import load_from_disk
from robust_multi_objective_decoding.constants import ProjectDir


class MultiObjectiveDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        labels: List[str],
        split: str = "train",
        train_test_val_split: List[float] = [0.8, 0.1, 0.1],
        apply_prompt_template=False,
        response_name: str = "response",
        prompt_name: str = "prompt",
        balance_dataset: bool = False,
        balancing_method: str = Literal["oversample", "downsample"],
        balancing_column: Optional[str] = None,
        single_objective: bool = False,
    ):
        """
        Initializes the MultiObjectiveDataset.
        Args:
            data_path (str): Path to the dataset file.
            labels (List[str]): List of label names.
            split (str, optional): Dataset split to use ('train', 'val', 'test', 'dev'). Defaults to 'train'.
            train_test_val_split (List[float], optional): Proportions for train, validation, and test splits. Must sum to 1. Defaults to [0.8, 0.1, 0.1].
            apply_prompt_template (bool, optional): Whether to apply a prompt template. Defaults to False.
            response_name (str, optional): Name of the response column. Defaults to 'response_0'.
            balance_dataset (bool, optional): Whether to balance the dataset. Defaults to False.
            balancing_method (Literal['oversample', 'downsample'], optional): Method to balance the dataset. Required if balance_dataset is True.
            balancing_column (Optional[str], optional): Column to use for balancing. Required if balance_dataset is True.
            single_objective (bool, optional): Return a single reward vector, not in a list
        Raises:
            AssertionError: If train_test_val_split does not sum to 1.
            ValueError: If balancing_column is not found in the dataset.
            NotImplementedError: If split is not 'train', 'val', 'test', or 'dev'.
        """

        # Check inputs
        assert (
            sum(train_test_val_split) == 1
        ), f"The train_test_val_split {train_test_val_split} split must sum to 1."

        if single_objective:
            assert (
                len(labels) == 1
            ), "single_objective cannot be True whilst num labels: {len(labels)} != 1"

        # Assign inputs
        self.labels = labels
        self.split = split
        self.apply_prompt_template = apply_prompt_template
        self.response_name = response_name
        self.single_objective = single_objective
        self.prompt_name = prompt_name

        # Load dataset and create split
        print("MultiObjectiveDataset: Loading data...")
        data = self.load_data(data_path)

        # Process and check the dataset:

        print("MultiObjectiveDataset: Checking labels...")
        self.validate_labels_length(data)

        print("MultiObjectiveDataset: Filtering responses...")
        data = self.filter_responses(data)
        assert len(data) > 0, "No responses left after filtering."

        # Balance the dataset:
        if balance_dataset:
            if balancing_column not in data.column_names:
                raise ValueError(
                    f"Balancing column {balancing_column} not found in dataset."
                )
            print("MultiObjectiveDataset: Balancing dataset...")
            data = self.balance_dataset(data, balancing_method, balancing_column)

        # Ensure duplicate prompts only appear in one split:
        df = data.to_pandas()
        unique_prompts = df.drop_duplicates(subset=self.prompt_name)

        train_prompts, val_prompts, test_prompts = self._get_train_test_val_prompts(
            unique_prompts[self.prompt_name], train_test_val_split
        )
        # Create data based on the split:
        if split == "train":
            data = df[df[self.prompt_name].isin(train_prompts)].reset_index(drop=True)
        elif split == "val":
            data = df[df[self.prompt_name].isin(val_prompts)].reset_index(drop=True)
        elif split == "test":
            data = df[df[self.prompt_name].isin(test_prompts)].reset_index(drop=True)
        elif split == "dev":
            prompts = pd.concat([train_prompts, val_prompts, test_prompts])
            data = df[df[self.prompt_name].isin(prompts)].reset_index(drop=True)
        else:
            raise NotImplementedError()

        self.data = datasets.Dataset.from_pandas(data)

    def get_train_test_val_indices(
        self, len_data: int, train_test_val_split: List[float]
    ):
        def get_train_test_val_indices(
            self, len_data: int, train_test_val_split: List[float]
        ):
            """
            Calculate the indices for splitting a dataset into training, validation, and test sets.
            Args:
                len_data (int): The total length of the dataset.
                train_test_val_split (List[float]): A list of three floats representing the proportions
                                                    of the dataset to be used for training, validation,
                                                    and testing, respectively. The values should sum to 1.
            Returns:
                tuple: A tuple containing three integers:
                    - upper_train (int): The index marking the end of the training set.
                    - upper_val (int): The index marking the end of the validation set.
                    - upper_test (int): The index marking the end of the test set (equal to len_data).
            """

        upper_train = int(len_data * train_test_val_split[0] // 1)
        upper_val = int(
            len_data * (train_test_val_split[0] + train_test_val_split[1]) // 1
        )
        upper_test = len_data

        return upper_train, upper_val, upper_test

    def _get_train_test_val_prompts(
        self, prompts, train_test_val_split: List[float]
    ) -> List[str]:
        """
        Split the prompt dataframe into train, test and validation sets.

        Parameters
        ----------
        prompts : pd.Series
            A pandas series of prompts.
        train_test_val_split : List[float]
            A list of floats that sum to 1. The first element is the train split, the second is the val split and the third is the test split.

        Returns
        -------
        List[str]
            The train, test and validation prompts
        """

        test_val_size = train_test_val_split[1] + train_test_val_split[2]

        train_prompts, test_val_prompts = train_test_split(
            prompts, test_size=test_val_size, random_state=42
        )

        val_prompts, test_prompts = train_test_split(
            test_val_prompts,
            test_size=train_test_val_split[2] / sum(train_test_val_split[1:]),
            random_state=42,
        )

        return train_prompts, val_prompts, test_prompts

    def load_data(self, data_path: str):
        """
        Load the dataset from the specified path.

        Written such that other data storage formats can be added later without
        adjusting the __init__ method. This approach makes unittesting easier as
        the load_data method can be patched.

        Parameters
        ----------
        data_path : str
            The path to the dataset.

        Returns
        -------
        datasets.Dataset
            The loaded dataset.
        """

        path = pathlib.Path(data_path)
        if not path.is_absolute():
            path = ProjectDir / path
        output = load_from_disk(path)
        return output

    def validate_labels_length(self, data: datasets.Dataset):
        """
        Check that the labels are the same length as the number of words in response_0.
        Only checks length for list-type labels, ignores scalar labels.
        """
        for label in self.labels:
            # Check if this label contains lists by examining first item
            first_item = data[0][label]
            if not isinstance(first_item, (list, tuple)):
                continue

            # Only process list-type labels
            for idx in range(len(data)):
                response = data[idx][self.response_name]
                labels = data[idx][label]
                assert (
                    len(response.split()) == len(labels)
                ), f"Response and labels length mismatch for label {label} at index {idx}."

    def filter_responses(self, data: datasets.Dataset):
        """
        Filter out responses with no words
        """
        return data.filter(lambda x: len(x[self.response_name].split()) > 1)

    def balance_dataset(
        self, data: datasets.Dataset, balancing_method: str, balancing_label: str
    ) -> datasets.Dataset:
        """
        Balances the dataset using the specified balancing method.
        Parameters:
        data (datasets.Dataset): The dataset to be balanced.
        balancing_method (str): The method to use for balancing the dataset.
                                Options are 'oversample' or 'downsample'.
        balancing_label (str): The label to balance the dataset on.
        Returns:
        datasets.Dataset: The balanced dataset.
        Raises:
        ValueError: If the balancing method is not recognized.
        """

        if balancing_method == "oversample":
            return self._oversample_minority_class(data, balancing_label)
        elif balancing_method == "downsample":
            return self._downsample_majority_class(data, balancing_label)
        else:
            raise ValueError(
                f'Balancing method {balancing_method} not recognized. Use "oversample" or "downsample".'
            )

    # Function to get label value handling both scalar and list cases
    def _get_label_value(self, example: Dict[str, Any], balancing_label: str) -> int:
        """
        Retrieve the value of a specified label from an example.
            example[balancing_label] is assumed to be a scalar or a list of scalars.
        """

        label = example[balancing_label]
        return label[-1] if isinstance(label, (list, tuple)) else label

    def _downsample_majority_class(
        self, data: datasets.Dataset, balancing_label: str
    ) -> datasets.Dataset:
        """
        Downsamples the majority class in the given dataset to balance the class distribution.
        Args:
            data (datasets.Dataset): The input dataset containing the data to be balanced.
            balancing_label (str): The label used to identify the class for balancing.
                The label is assumed to be binary with values 0 and 1.
        Returns:
            datasets.Dataset: A new dataset with the majority class downsampled to match the size of the minority class.
        """

        # Split data by class
        class_1 = data.filter(lambda x: self._get_label_value(x, balancing_label) == 1)
        class_2 = data.filter(lambda x: self._get_label_value(x, balancing_label) == 0)

        # Identify majority and minority classes
        if len(class_1) > len(class_2):
            majority_class = class_1
            minority_class = class_2
        else:
            majority_class = class_2
            minority_class = class_1

        # Calculate target size for downsampling
        target_size = len(minority_class)

        # Randomly sample from majority class
        keep_indices = random.sample(range(len(majority_class)), target_size)
        downsampled_majority = majority_class.select(keep_indices)

        # Combine downsampled majority with minority class
        balanced_dataset = datasets.concatenate_datasets(
            [downsampled_majority, minority_class]
        )

        return balanced_dataset.shuffle(seed=42)

    def _oversample_minority_class(
        self, data: datasets.Dataset, balancing_label: str
    ) -> datasets.Dataset:
        """
        Oversamples the minority class in the given dataset to balance the class distribution.
        Args:
            data (datasets.Dataset): The dataset containing the data to be balanced.
            balancing_label (str): The label used to determine the class distribution.
                The label is assumed to be binary with values 0 and 1.
        Returns:
            datasets.Dataset: A new dataset with the minority class oversampled to match the majority class size.
        """

        # Split the two label classes
        class_1_data = data.filter(
            lambda x: self._get_label_value(x, balancing_label) == 1
        )
        class_0_data = data.filter(
            lambda x: self._get_label_value(x, balancing_label) == 0
        )

        # Determine majority and minority sets
        if len(class_1_data) > len(class_0_data):
            majority = class_1_data
            minority = class_0_data
        else:
            majority = class_0_data
            minority = class_1_data

        # Calculate how many additional samples we need
        samples_needed = len(majority) - len(minority)

        # If minority set is too small, repeat it entirely multiple times and then sample remainder
        full_repeats = samples_needed // len(minority)
        remainder = samples_needed % len(minority)

        # Create oversampled dataset
        oversampled = datasets.concatenate_datasets([minority] * (full_repeats + 1))
        if remainder > 0:
            remainder_indices = random.sample(range(len(minority)), remainder)
            remainder_samples = minority.select(remainder_indices)
            oversampled = datasets.concatenate_datasets(
                [oversampled, remainder_samples]
            )

        # Combine with majority class and shuffle
        balanced_dataset = datasets.concatenate_datasets([majority, oversampled])
        return balanced_dataset.shuffle(seed=42)

    def __getitem__(self, idx):
        if self.single_objective:
            # Return a single objective
            labels_list = self.data[idx][self.labels[0]]
        else:
            labels_list = [self.data[idx][label] for label in self.labels]

        data = self.data[idx][self.response_name]
        prompt = self.data[idx][self.prompt_name]

        # In the test case we want to append Prompt and Answer to the output:
        # TODO: we might want to investigate training on this for better prompting
        if self.apply_prompt_template:
            prompt = f"Prompt:\n\n{prompt}\n\nAnswer:\n\n"

        return idx, prompt, data, labels_list

    def __len__(self):
        return len(self.data)

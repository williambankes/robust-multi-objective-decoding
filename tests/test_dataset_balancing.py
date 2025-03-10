import pytest
import torch
import datasets
from robust_multi_objective_decoding.data.balanced_dataloader import BalancedDataLoader
from robust_multi_objective_decoding.data.safety_datasets import PKUSafeRLHFDataset


class MockPKUSafeRLHFDataset(PKUSafeRLHFDataset):
    """Mock dataset class that overrides loading and initialization"""
    def __init__(self, data=None, balance_dataset=False):
        self.balance_dataset = balance_dataset
        if data is None:
            # Default imbalanced data
            self.data = datasets.Dataset.from_dict({
                'prompt': ['p1', 'p2', 'p3', 'p4', 'p5'],
                'response_0': ['r1', 'r2', 'r3', 'r4', 'r5'],
                'combined_harm_label': [
                    [0, 0, 0],  # unsafe
                    [0, 0, 0],  # unsafe
                    [0, 0, 0],  # unsafe
                    [0, 0, 1],  # safe
                    [0, 0, 1],  # safe
                ]
            })
        else:
            self.data = data
            
        if self.balance_dataset:
            self.data = self.oversample_minority_class(self.data)
    
    def __getitem__(self, idx):
        return {
            'prompt': self.data[idx]['prompt'],
            'response_0': self.data[idx]['response_0'],
            'combined_harm_label': self.data[idx]['combined_harm_label']
        }
    
    def __len__(self):
        return len(self.data)


def test_dataset_oversampling():
    """Test that dataset-level oversampling properly balances classes"""
    # Test with imbalanced data (3 unsafe, 2 safe)
    dataset = MockPKUSafeRLHFDataset(balance_dataset=True)
    
    # Count safe and unsafe samples after balancing
    safe_count = sum(1 for i in range(len(dataset)) 
                    if dataset.data[i]['combined_harm_label'][-1] == 1)
    unsafe_count = sum(1 for i in range(len(dataset)) 
                      if dataset.data[i]['combined_harm_label'][-1] == 0)
    
    # After balancing, should have equal numbers
    assert safe_count == unsafe_count, \
        f"Unequal classes after oversampling: safe={safe_count}, unsafe={unsafe_count}"
    
    # Test with different imbalance ratio
    more_imbalanced_data = datasets.Dataset.from_dict({
        'prompt': ['p1', 'p2', 'p3', 'p4'],
        'response_0': ['r1', 'r2', 'r3', 'r4'],
        'combined_harm_label': [
            [0, 0, 0],  # unsafe
            [0, 0, 0],  # unsafe
            [0, 0, 0],  # unsafe
            [0, 0, 1],  # safe
        ]
    })
    
    dataset = MockPKUSafeRLHFDataset(data=more_imbalanced_data, balance_dataset=True)
    
    safe_count = sum(1 for i in range(len(dataset)) 
                    if dataset.data[i]['combined_harm_label'][-1] == 1)
    unsafe_count = sum(1 for i in range(len(dataset)) 
                      if dataset.data[i]['combined_harm_label'][-1] == 0)
    
    assert safe_count == unsafe_count, \
        f"Unequal classes after oversampling: safe={safe_count}, unsafe={unsafe_count}"
    
    # Verify oversampled data maintains original samples
    original_safe = [i for i in range(len(more_imbalanced_data)) 
                    if more_imbalanced_data[i]['combined_harm_label'][-1] == 1]
    
    # Check all original safe samples are in oversampled dataset
    for idx in original_safe:
        orig_sample = more_imbalanced_data[idx]
        found = False
        for i in range(len(dataset)):
            if (dataset.data[i]['prompt'] == orig_sample['prompt'] and 
                dataset.data[i]['response_0'] == orig_sample['response_0']):
                found = True
                break
        assert found, f"Original safe sample at index {idx} not found in oversampled dataset"


def test_balanced_dataloader():
    """Test that the BalancedDataLoader properly balances safe and unsafe samples"""
    # Create dataset with imbalanced data
    dataset = MockPKUSafeRLHFDataset(balance_dataset=False)
    dataloader = BalancedDataLoader(dataset, batch_size=4, drop_last=True)

    print(dataset[0])
    
    # Collect samples over multiple iterations to check balance
    samples = []
    for _ in range(100):
        batch = next(iter(dataloader))
        print(_, batch['combined_harm_label'])
        print(type(batch['combined_harm_label']))
        print(_, batch['prompt'])
        samples.extend([batch['combined_harm_label'][-1].tolist()])
    
    # Convert to tensor for easier counting
    samples = torch.tensor(samples)
    safe_count = (samples == 1).sum().item()
    unsafe_count = (samples == 0).sum().item()
    
    # Check that safe and unsafe samples are roughly balanced
    ratio = safe_count / unsafe_count
    print(f'Dataset size: {len(dataset)}')
    print(f'Ratio: {ratio}, safe: {safe_count}, unsafe: {unsafe_count}')
    assert 0.8 < ratio < 1.2, f"Imbalanced sampling ratio: {ratio}"
    
    # Test with different imbalance ratio
    more_imbalanced_data = datasets.Dataset.from_dict({
        'prompt': ['p1', 'p2', 'p3', 'p4'],
        'response_0': ['r1', 'r2', 'r3', 'r4'],
        'combined_harm_label': [
            [0, 0, 0],  # unsafe
            [0, 0, 0],  # unsafe
            [0, 0, 0],  # unsafe
            [0, 0, 1],  # safe
        ]
    })
    
    dataset = MockPKUSafeRLHFDataset(data=more_imbalanced_data, balance_dataset=False)
    dataloader = BalancedDataLoader(dataset, batch_size=4)
    
    samples = []
    for _ in range(100):
        batch = next(iter(dataloader))
        samples.extend([batch['combined_harm_label'][-1].tolist()])
    
    samples = torch.tensor(samples)
    safe_count = (samples == 1).sum().item()
    unsafe_count = (samples == 0).sum().item()
    
    # Check balance with new ratio
    ratio = safe_count / unsafe_count
    assert 0.8 < ratio < 1.2, f"Imbalanced sampling ratio: {ratio}"


def test_combined_balancing():
    """Test that dataset oversampling and dataloader balancing work together"""
    # Create dataset with severe imbalance
    imbalanced_data = datasets.Dataset.from_dict({
        'prompt': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6'],
        'response_0': ['r1', 'r2', 'r3', 'r4', 'r5', 'r6'],
        'combined_harm_label': [
            [0, 0, 0],  # unsafe
            [0, 0, 0],  # unsafe
            [0, 0, 0],  # unsafe
            [0, 0, 0],  # unsafe
            [0, 0, 0],  # unsafe
            [0, 0, 1],  # safe
        ]
    })
    
    # Test with both balancing methods
    dataset = MockPKUSafeRLHFDataset(data=imbalanced_data, balance_dataset=True)
    dataloader = BalancedDataLoader(dataset, batch_size=4)
    
    # Verify dataset is balanced
    safe_count_dataset = sum(1 for i in range(len(dataset)) 
                           if dataset.data[i]['combined_harm_label'][-1] == 1)
    unsafe_count_dataset = sum(1 for i in range(len(dataset)) 
                             if dataset.data[i]['combined_harm_label'][-1] == 0)
    assert safe_count_dataset == unsafe_count_dataset, \
        "Dataset not balanced after oversampling"
    
    # Verify dataloader sampling is balanced
    samples = []
    for _ in range(100):
        batch = next(iter(dataloader))
        samples.extend([batch['combined_harm_label'][-1].tolist()])
    
    samples = torch.tensor(samples)
    safe_count = (samples == 1).sum().item()
    unsafe_count = (samples == 0).sum().item()
    
    ratio = safe_count / unsafe_count
    assert 0.8 < ratio < 1.2, f"Imbalanced sampling ratio: {ratio}"

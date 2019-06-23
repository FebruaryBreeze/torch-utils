import unittest

import torch.nn as nn

import torch_utils


class MockModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, 3)
        self.classifier = nn.Linear(16, 1000)
        self.bn = nn.BatchNorm2d(16)

    def forward(self, x):
        raise NotImplementedError


class MyTestCase(unittest.TestCase):
    def test_initialize_weights(self):
        model = MockModule()
        torch_utils.initialize_weights(model)
        self.assertTrue(model.bn.weight[0].item() == 1.0)


if __name__ == '__main__':
    unittest.main()

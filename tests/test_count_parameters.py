import unittest

import torch.nn as nn

import torch_utils


class MockModule(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        raise NotImplementedError


class MyTestCase(unittest.TestCase):
    def test_count_parameters(self):
        c = 16
        model = MockModule(c)
        model.eval()

        num_params, num_buffers = torch_utils.count_parameters(model)
        self.assertEqual(num_params, c * 3 * 3 * c + c * 2)
        self.assertEqual(num_buffers, c * 2 + 1)


if __name__ == '__main__':
    unittest.main()

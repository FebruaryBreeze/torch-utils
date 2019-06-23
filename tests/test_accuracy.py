import unittest

import torch

import torch_utils


class MyTestCase(unittest.TestCase):
    def test_accuracy(self):
        output = torch.Tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
        target = torch.Tensor([3, 2]).long()

        top_1, top_3 = torch_utils.accuracy(output=output, target=target, top_k=(1, 3))
        self.assertAlmostEqual(top_1.item(), 50)
        self.assertAlmostEqual(top_3.item(), 100)


if __name__ == '__main__':
    unittest.main()

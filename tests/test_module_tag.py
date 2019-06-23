import unittest

import torch.nn as nn

import torch_utils


class Green(torch_utils.ModuleTag):
    pass


class White(torch_utils.ModuleTag):
    pass


class MockModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.green_conv = Green(nn.Conv2d(1, 1, kernel_size=3))
        self.white_conv = White(nn.Conv2d(1, 1, kernel_size=5))
        self.green_bn = Green(nn.BatchNorm2d(16))
        self.white_bn = White(nn.BatchNorm2d(16))
        self.white_module = White(SubModule())

    def forward(self):
        raise NotImplementedError


class SubModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=5)
        self.sub_green_conv = Green(nn.Conv2d(1, 1, kernel_size=3))

    def forward(self):
        raise NotImplementedError


class MyTestCase(unittest.TestCase):
    def test_state_dict_with_tag(self):
        model = MockModule()

        state_dict = torch_utils.state_dict_with_tag(model)
        for key, (param, tag) in state_dict.items():
            if 'sub_green' in key:
                self.assertIs(tag, Green)
            elif key.startswith('white'):
                self.assertIs(tag, White)
            elif key.startswith('green'):
                self.assertIs(tag, Green)
            else:
                raise ValueError

        self.assertEqual(state_dict.keys(), model.state_dict().keys())

    def test_named_parameters_with_tag(self):
        model = MockModule()

        for key, param, tag in torch_utils.named_parameters_with_tag(module=model):
            if 'sub_green' in key:
                self.assertIs(tag, Green)
            elif key.startswith('white'):
                self.assertIs(tag, White)
            elif key.startswith('green'):
                self.assertIs(tag, Green)
            else:
                raise ValueError


if __name__ == '__main__':
    unittest.main()

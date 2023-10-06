import unittest
from xas.dataset import XASDataModule
import torch


class TestXASDataModule(unittest.TestCase):
    def test_load_dataset(self):
        data_module = XASDataModule()
        self.assertEqual(data_module.train_dataset[0][0].shape, torch.Size([64]))
        self.assertEqual(data_module.train_dataset[0][1].shape, torch.Size([200]))


if __name__ == "__main__":
    unittest.main()

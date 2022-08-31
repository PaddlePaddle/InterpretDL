import unittest
import paddle
import numpy as np
import shutil
from paddle.vision.models.mobilenetv2 import mobilenet_v2

import interpretdl as it


class IndexedDataset(paddle.io.Dataset):
    """
    Modify output of getitem.
    
    :param :`paddle.io.Dataset`: base_dataset: Dataset to wrap

    """
    def __init__(self,
                 base_dataset: object):
        super().__init__()
        self.dataset = base_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        return index, (input/255.0).astype(np.float32), int(target)


class TestTD(unittest.TestCase):

    def test_generator(self):
        from paddle.vision.transforms import Transpose
        from paddle.vision.datasets import Cifar10
        from paddle.io import Dataset, Subset
        train_dataset = IndexedDataset(Cifar10(mode='train', transform=Transpose()))
        train_dataset = paddle.io.Subset(dataset=train_dataset, indices=list(range(50)))
        paddle_model = mobilenet_v2(pretrained=False, scale=0.5)

        train_loader = paddle.io.DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=4)
        scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=0.01, milestones=[10, 15], gamma=0.1)
        optimizer = paddle.optimizer.Momentum(learning_rate=scheduler, parameters=paddle_model.parameters())
        td = it.TrainingDynamics(paddle_model)
        training_dynamics = td.generator(train_loader, optimizer, epochs=10)
       
        assigned_targets = []
        for i in range(50):
            t = train_dataset[i][2]
            assigned_targets.append(t)
        td.save(training_dynamics,assigned_targets,label_flip=None,save_path="assets")

    def test_zBHDF(self):
        BHDF = it.BHDFInterpreter(device='gpu:0')
        order, predictions = BHDF.interpret(training_dynamics_path="assets/training_dynamics.npz")
        # the predictions are not correct here.


if __name__ == '__main__':
    unittest.main()
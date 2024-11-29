from __future__ import annotations

from torchvision.datasets import FashionMNIST

from project.datamodules.image_classification.mnist import MNISTDataModule


class FashionMNISTDataModule(MNISTDataModule):
    """
    .. figure:: https://storage.googleapis.com/kaggle-datasets-images/2243/3791/9384af51de8baa77f6320901f53bd26b/dataset-cover.png
        :width: 400
        :alt: Fashion MNIST

    Specs:
        - 10 classes (1 per type)
        - Each image is (1 x 28 x 28)

    Standard FashionMNIST, train, val, test splits and transforms

    Transforms::

        mnist_transforms = transform_lib.Compose([
            transform_lib.ToTensor()
        ])

    Example::

        from pl_bolts.datamodules import FashionMNISTDataModule

        dm = FashionMNISTDataModule('.')
        model = LitModel()

        Trainer().fit(model, datamodule=dm)
    """

    name = "fashion_mnist"
    dataset_cls = FashionMNIST

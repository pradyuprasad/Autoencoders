import torchvision
from torchvision.transforms import ToTensor
from display_image import plot_image

train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=ToTensor()
)


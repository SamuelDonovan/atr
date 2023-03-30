# From pytorch
import torch


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # Convolutional layers
        self.convolution_1 = torch.nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, padding=1
        )
        self.convolution_2 = torch.nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=5, padding=2
        )
        self.convolution_3 = torch.nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, padding=2
        )
        self.convolution_4 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, padding=2
        )

        # Batch normalization layers
        self.batch_norm_1 = torch.nn.BatchNorm2d(num_features=8)
        self.batch_norm_2 = torch.nn.BatchNorm2d(num_features=16)
        self.batch_norm_3 = torch.nn.BatchNorm2d(num_features=32)
        self.batch_norm_4 = torch.nn.BatchNorm2d(num_features=64)

        # Dropout layers
        self.dropout_1 = torch.nn.Dropout(p=0.1)
        self.dropout_2 = torch.nn.Dropout(p=0.2)
        self.dropout_3 = torch.nn.Dropout(p=0.3)
        self.dropout_4 = torch.nn.Dropout(p=0.4)

        # Fully connected layers
        self.fully_connected_1 = torch.nn.Linear(
            in_features=64 * 4 * 4, out_features=256
        )
        self.fully_connected_2 = torch.nn.Linear(in_features=256, out_features=128)
        self.fully_connected_3 = torch.nn.Linear(in_features=128, out_features=43)

    def forward(self, x):
        x = torch.nn.functional.relu(self.batch_norm_1(self.convolution_1(x)))
        x = self.dropout_1(x)
        x = torch.nn.functional.relu(self.batch_norm_2(self.convolution_2(x)))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2)
        x = self.dropout_2(x)
        x = torch.nn.functional.relu(self.batch_norm_3(self.convolution_3(x)))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2)
        x = self.dropout_3(x)
        x = torch.nn.functional.relu(self.batch_norm_4(self.convolution_4(x)))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2)
        x = self.dropout_4(x)
        x = x.view(-1, 64 * 4 * 4)
        x = torch.nn.functional.relu(self.fully_connected_1(x))
        x = torch.nn.functional.relu(self.fully_connected_2(x))
        x = self.fully_connected_3(x)
        return x

import torch.nn as nn
import torch.nn.functional as F


###############################################################################
# Just a simple MLP with one hidden layer.                                    #
###############################################################################
class CalvisMLP(nn.Module):
    """
    If debug is True information about the network is printed.
    """

    def __init__(self, debug=False):
        super().__init__()

        self.debug = debug

        self.fc1 = nn.Linear(50 * 50 * 1, 120)
        self.fc2 = nn.Linear(120, 1)

    def forward(self, x):
        if self.debug:
            print("Shape of the input: ")
            print(x.shape)
            print("\n")

        # Flatten the tensor to input it to the FC Layer:
        x = x.view(-1, self.num_flat_features(x))
        if self.debug:
            print("Shape of the data before entering the fully connected: ")
            print(x.shape)
            print("\n")

        x = F.relu(self.fc1(x))
        if self.debug:
            print("Shape of the data after the hidden layer: ")
            print(x.shape)
            print("\n")

        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


###############################################################################
# MLP with 2 hidden layers.                                                   #
###############################################################################
class Calvis2HiddenLayerMLP(nn.Module):
    """
    If debug is True information about the network is printed.
    """

    def __init__(self, debug=False):
        super().__init__()

        self.debug = debug

        self.fc1 = nn.Linear(50 * 50 * 1, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        if self.debug:
            print("Shape of the input: ")
            print(x.shape)
            print("\n")

        # Flatten the tensor to input it to the FC Layer:
        x = x.view(-1, self.num_flat_features(x))
        if self.debug:
            print("Shape of the data before entering the fully connected: ")
            print(x.shape)
            print("\n")

        x = F.relu(self.fc1(x))
        if self.debug:
            print("Shape of the data after the hidden layer: ")
            print(x.shape)
            print("\n")
        x = F.relu(self.fc2(x))
        if self.debug:
            print("Shape of the data after ouput layer: ")
            print(x.shape)
            print("\n")

        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


###############################################################################
# CNN with one convolutional layer (without pooling) and two FC layers.       #
###############################################################################
class Calvis1CNN(nn.Module):
    """
    If debug is True information about the network is printed.
    """

    def __init__(self, debug=False):
        super().__init__()

        self.debug = debug
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.fc1 = nn.Linear(46 * 46 * 3, 120)
        self.fc2 = nn.Linear(120, 3)

    def forward(self, x):
        if self.debug:
            print("Shape of the input: ")
            print(x.shape)
            print("\n")

        x = F.relu(self.conv1(x))
        if self.debug:
            print("Shape of the data after the first convolution: ")
            print(x.shape)
            print("\n")

        # Flatten the tensor to input it to the FC Layer:
        x = x.view(-1, self.num_flat_features(x))
        if self.debug:
            print("Shape of the data before entering the fully connected: ")
            print(x.shape)
            print("\n")

        x = F.relu(self.fc1(x))
        if self.debug:
            print("Shape of the data after the hidden layer: ")
            print(x.shape)
            print("\n")

        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


###############################################################################
# CNN with two convolutional layer, maxpooling and two FC layers.             #
###############################################################################
class Calvis2CNN(nn.Module):
    """
    If debug is True information about the network is printed.
    """

    def __init__(self, debug=False):
        super().__init__()

        self.debug = debug
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.bn = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3, 7, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(9 * 9 * 7, 84)
        self.fc2 = nn.Linear(84, 3)

    def forward(self, x):
        if self.debug:
            print("Shape of the input: ")
            print(x.shape)
            print("\n")

        x = self.bn(F.relu(self.conv1(x)))
        if self.debug:
            print("Shape of the data after the first convolution: ")
            print(x.shape)
            print("\n")

        x = self.pool(x)
        if self.debug:
            print("Shape of the data after first pooling: ")
            print(x.shape)
            print("\n")

        x = F.relu(self.conv2(x))
        if self.debug:
            print("Shape of the data after the second convolution: ")
            print(x.shape)
            print("\n")

        x = self.pool(x)
        if self.debug:
            print("Shape of the data after second pooling: ")
            print(x.shape)
            print("\n")

        x = x.view(-1, self.num_flat_features(x))
        if self.debug:
            print("Shape of the data before entering the fully connected: ")
            print(x.shape)
            print("\n")

        x = F.relu(self.fc1(x))
        if self.debug:
            print("Shape of the data after the hidden layer: ")
            print(x.shape)
            print("\n")

        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


###############################################################################
# CNN with 1 convolutional layer, maxpooling and 1 fc layers.                #
###############################################################################
class CalvisVanillaCNNet(nn.Module):
    """
    If debug is True information about the network is printed.
    """

    def __init__(self, debug=False):
        # Call constructor on nn.Module
        super().__init__()
        self.debug = debug

        self.conv = nn.Conv2d(1, 3, 5)
        self.bn = nn.BatchNorm2d(3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(in_features=3 * 98 * 98, out_features=3)

    def forward(self, x):
        if self.debug:
            print("Shape of the input: ")
            print(x.shape)
            print("\n")

        # First convolution and non linearity
        # At this point the data has shape ([5, 3, 196, 196]) because the batch
        # size is 5, there are 3 output channels (defined on the conv layer
        # before). Additionaly, you can slide a 5x5 filter 'only' 196 times
        # over a 200x200 image (asuming slide = 1 and padding = 1) and
        # everytime you slide the filter a value is computed, therefore the
        # dimensions are 196 x 196.
        x = F.relu(self.bn(self.conv(x)))
        if self.debug:
            print("Shape of the data after the first convolution: ")
            print(x.shape)
            print("\n")

        # Max pooling over a (2, 2) window
        # Both W and H dimensions are reduced by two.
        # At this point the data has shape ([5, 3, 98, 98])
        x = self.pool(x)
        if self.debug:
            print("Shape of the data after first pooling: ")
            print(x.shape)
            print("\n")

        # Flatten the tensor to input it to the FC Layer:
        # At this point the data has shape ([5, 3, 98, 98])
        x = x.view(-1, self.num_flat_features(x))
        if self.debug:
            print("Shape of the data before entering the fully connected: ")
            print(x.shape)
            print("\n")

        x = self.fc(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":
    net = CalvisVanillaCNNet()
    print(net)

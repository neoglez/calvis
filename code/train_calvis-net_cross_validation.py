import numpy as np
from torchvision import transforms
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from CalvisNet import CalvisVanillaCNNet
from CalvisDataset import CalvisFairCMU2DDataset
from CalvisTransform import TwoDToTensor
import os

#############################################################################


def train_ann(ann, device, dataloader, criterion, epochs):
    ann = ann.to(device)
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dataloader):
            # print(i)
            # print(data)
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = (
                data["image"].to(device),
                data["annotations"]["human_dimensions"].to(device),
            )
            

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = ann(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:  # print every 20 mini-batches
                print(
                    "[epoch %d, pattern number %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / 20)
                )
                running_loss = 0.0

    print("Finished Training")


if __name__ == "__main__":
    torch.manual_seed(1)  # reproducible
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    ###########################################################################
    projectDir = os.path.abspath("./../CALVIS/")
    rootDir = os.path.join(projectDir, "dataset/cmu/")
    imgDir = os.path.join(rootDir, "synthetic_images/200x200/")

    ###########################################################################
    dataset = CalvisFairCMU2DDataset(rootDir, imgDir, transform=TwoDToTensor())

    # Set the k-fold
    k = 3
    # Split the indices into k mutually exclusive subsets $\mathcal{D}_i$
    indices = range(len(dataset))
    partitions = kf = KFold(n_splits=k, random_state=None, shuffle=True)
    fold = 0
    loss_function = torch.nn.L1Loss()
    # The error vector contains errors $e_i$ for every pattern $z^{(i)}$.
    # The size of this vector in a multiple M task scenario with continuos
    # output (multivariate regression) for a dataset with N pattern is (M x N).
    number_of_tasks = 3
    # error_vector = np.zeros((number_of_tasks, len(dataset)))

    for train_index, test_index in kf.split(indices):
        fold += 1
        print("Training in fold number:", fold)

        # Define the network for this fold. It is a kind of weight reset.
        # In more complex scenario we could use different ANN for every fold.
        # For example, assuming there is a function taking an integer and
        # returning a network we could make net = get_network_for_fold(fold)
        net = CalvisVanillaCNNet()

        # print(net)  # net architecture

        # We globaly define the hyperparamers but they could be paramerters
        # of the training algo.
        epochs = 20
        optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
        criterion = torch.nn.L1Loss()

        current_training_d_without_d_i = SubsetRandomSampler(
            indices=train_index
        )

        current_training_d_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            sampler=current_training_d_without_d_i,
            batch_sampler=None,
            num_workers=0,
            collate_fn=None,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            multiprocessing_context=None,
        )

        current_d_i = SubsetRandomSampler(indices=test_index)

        current_d_i_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            sampler=current_d_i,
            batch_sampler=None,
            num_workers=0,
            collate_fn=None,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            multiprocessing_context=None,
        )
        # train CNN
        # $f_i$ is the learning algorithm. In this case, is the ANN with the
        # "best parameters" according to the loss function used inside the
        # training loop. Note that network architecture, loss function
        # (criterion) and number of iterations (epochs) remain constant.
        # However, these paramters could be changed to perform a model
        # selection/evaluation.
        train_ann(
            ann=net,
            device=device,
            dataloader=current_training_d_loader,
            criterion=criterion,
            epochs=epochs,
        )
        
        f_i = net.to()

        # Calculate loss of the trained model output and the data elements of
        # the current partition. Note that we could use now a different loss
        # function than the one used to train the network itself. Nevertheless,
        # I use the same here (L1 loss).
        current_loss = 0.0
        print("Validating in fold number:", fold)
        for i, data in enumerate(current_d_i_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = (
                data["image"].to(device),
                data["annotations"]["human_dimensions"].to(device)
            )

            # only forward because we are evaluating
            outputs = net(inputs)
            loss = loss_function(outputs, labels)

            # print statistics
            current_loss = loss.item()
            # i is the pattern index but numpy is zero-indexed
            #error_vector[i] = current_loss
            if i % 20 == 19:  # print every 20 mini-batches
                print("[Fold %d, %5d] loss: %.3f" % (fold, i, current_loss))


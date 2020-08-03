import numpy as np
import argparse
import pandas as pd
import torchvision.models as models
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
import torch.nn as nn
import torch.optim as optim

from data.xray_dataset import XrayImageDataset
from data.model import XrayModel


def main():
    device = (torch.device("cuda") if options.cuda and torch.cuda.is_available() else "cpu")

    # Open training dataset file
    train_df = pd.read_csv('data/boneage-training-dataset.csv', index_col=False)

    # Calculate mean and standard deviation
    mean_boneage = train_df['boneage'].mean()
    std_boneage = train_df['boneage'].std()

    # Declare preprocessing operations - Resize and convert to tensors
    preprocess_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    # Load datasets using custom Dataset instance
    training_dataset = XrayImageDataset(
        csv_file='data/training_data.csv',
        root_dir='data/training_images/',
        transform=preprocess_transform
    )

    testing_dataset = XrayImageDataset(
        csv_file='data/eval_data.csv',
        root_dir='data/training_images/',
        transform=preprocess_transform
    )

    # Define dataloaders using dataset of transformed images
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    testing_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print('Initialising Model')
    if options.startingModel:  # Load selected model if restarting training on intermediate model
        my_model = torch.load(options.startingModel)
    else:
        my_model = XrayModel()

    # Send model to GPU
    my_model.to(device)

    criterion = nn.MSELoss()  # Declare MSE and the minimisation goal for my network
    optimizer = optim.Adam(my_model.parameters(), lr=learn_rate)  # Using Adam to optimise

    def train_model(epoch):
        my_model.train()  # Set model to training mode
        total_mae = 0  # Running total of MAE - used to calculate average MAE
        for i_batch, sample_batched in enumerate(training_dataloader):
            # Move variables to selected device
            image = sample_batched['image'].to(device)
            age = sample_batched['bone_age_z'].type(torch.float32).to(device)
            gender = sample_batched['gender'].to(device)

            optimizer.zero_grad()  # clear gradients

            guess = my_model(image, gender)  # perform inference

            MAE = mae_to_months(age.cpu().detach().numpy(), guess.cpu().detach().numpy())  # Calculate MAE

            loss = criterion(guess, age)  # Calculate loss

            loss.backward()  # Compute gradients

            optimizer.step()  # Update gradients

            total_mae += MAE  # Keep running total of MAE

            # Output training progress along with loss at each training step and the MAE for that batch
            print("Epoch[{}] - Processed Batch[{}/{}] - MSE Loss: {:.4f} - MAE: {:.4f}".format(epoch, i_batch + 1,
                                                                                               len(training_dataloader),
                                                                                               loss.item(), MAE))
        return total_mae

    def test_model():
        my_model.eval()  # Set model to evaluation mode - so the parameter weights cannot be adjusted
        mse = 0  # Running totals for MSE and MAE
        total_mae = 0
        with torch.no_grad():
            for batch in testing_dataloader:
                # Move variables to selected device
                image = batch['image'].to(device)
                age = batch['bone_age_z'].type(torch.float32).to(device)
                gender = batch['gender'].to(device)

                guess = my_model(image, gender)  # perform inference

                MAE = mae_to_months(age.cpu().detach().numpy(), guess.cpu().detach().numpy())  # calculate MAE

                total_mae += MAE  # Keep running total of MAE

                mse += criterion(guess, age)  # Keep running total of MSE

        print("-TEST- Average MSE: {:.4f}".format(mse / len(training_dataloader)))  # Output average MSE
        return total_mae

    def save_model(epoch):
        model_save_path = "trained_models/model_{}.pth".format(epoch)  # Save model with epoch number in the name
        torch.save(my_model, model_save_path)  # Save model
        print("MODEL SAVED - {}".format(model_save_path))  # Print where model is saved

    def mae_to_months(x, y):  # Calculate mean absolute error and convert from z-score to months.

        return mean_absolute_error((std_boneage * x + mean_boneage),  # Uses mean_absolute_error from sklearn.metrics
                                   (std_boneage * y + mean_boneage))

    for epoch in range(1, n_epochs + 1):
        mae_train_total = train_model(epoch)
        mae_test_total = test_model()

        average_mae = mae_train_total / len(training_dataloader)
        average_test_mae = mae_test_total / len(testing_dataloader)

        print("Training MAE: {} -- Testing MAE: {}".format(average_mae, average_test_mae))

        save_model(epoch)  # Save model every epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hand xray age estimator')
    parser.add_argument('--batchSize', type=int, default=8, help="Training batch size (8 ~= 6GB vram)")
    parser.add_argument('--testBatchSize', type=int, default=32, help='Testing batch size - 32 default')
    parser.add_argument('--epochCount', type=int, default=40, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate - 0.01 default')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use CUDA?')
    parser.add_argument('--startingModel', type=str, help='Model path to start training')

    options = parser.parse_args()

    print(options)

    batch_size = options.batchSize
    n_epochs = options.epochCount
    learn_rate = options.lr

    main()

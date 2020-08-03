import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import mean_absolute_error
from data.xray_dataset import XrayImageDataset
import matplotlib.pyplot as plt




def mae_to_months(x, y):
    # Calculate mean absolute error and convert from z-score to months.
    return mean_absolute_error((std_boneage * x + mean_boneage), (std_boneage * y + mean_boneage))


def main():
    # Declare evaluation dataset
    eval_dataset = XrayImageDataset(
        csv_file='data/testing_data.csv',
        root_dir='data/training_images/',
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ]))

    # Declare DataLoader for evaluation data
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create empty array to store average MAE after each model is tested
    MAE = np.zeros(models_count)

    # Test model at each epoch
    for i in range(models_count):
        model_path = "trained_models/model_{}.pth".format(i+1)

        print("Evaluating model at epoch {}".format(i+1))

        my_model = torch.load(model_path)  # Load model

        with torch.no_grad():  # Do not compute gradients (No need to as not training the model)
            mae_total = 0
            for batch in eval_dataloader:
                # Move variables to GPU
                image = batch['image'].to(device)
                age = batch['bone_age_z'].type(torch.float32).to(device)
                gender = batch['gender'].to(device)

                predict = my_model(image, gender)  # Perform inference

                # Running total of MAE
                mae_total += mae_to_months(age.cpu().detach().numpy(), predict.cpu().detach().numpy())

        avg_MAE = mae_total / len(eval_dataloader)  # Calculate average MAE
        MAE[i] = avg_MAE  # Add average MAE to array
        print("Average MAE for model {}: {}".format(i+1, avg_MAE))

    # Display graph showing error per epoch
    plt.plot(range(1, models_count+1), MAE)
    plt.xticks(range(1, models_count+1))
    plt.title('Model  Error')
    plt.ylabel('Error (MAE)')
    plt.xlabel('Epoch')

    plt.show()


if __name__ == '__main__':
    models_path = "trained_models/"  # Maybe make this find highest epoch model in folder
    batch_size = 32

    list = os.listdir(models_path)  # Find all files in trained_model folder
    models_count = len(list)  # Count number of models in file
    print("Number of models: {}".format(models_count))  # Display number of models found

    # Use CUDA if available
    device = (torch.device("cuda") if torch.cuda.is_available() else "cpu")

    # Load training dataset
    train_df = pd.read_csv('data/boneage-training-dataset.csv', index_col=False)

    # Calculate mean and standard deviation of ages in dataset - for converting z-score
    mean_boneage = train_df['boneage'].mean()
    std_boneage = train_df['boneage'].std()


    main()

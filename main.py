import torch
from PIL import Image, ImageOps
import pandas as pd
import os
import matplotlib.image as mpimg
import torchvision.transforms as transforms
import sys
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from gui import Ui_MainWindow
import ntpath


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    # Setting the default image num which is set when a image is select and passed to estimate_boneage.
    image_num = 0

    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent=parent)
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon('data/skeleton.ico'))
        self.predict_button.clicked.connect(self.estimate_boneage)
        self.select_image_button.clicked.connect(self.select_image_file)

    def select_image_file(self):
        # Opens the file picker and sets the returned path to a variable.
        image_file = QFileDialog.getOpenFileName(self, 'Single File', 'data/eval_images')

        # Updates the pixmap with the selected image.
        self.image_pixmap.setPixmap(QtGui.QPixmap(str(image_file[0])))

        # Updates the label with the file path.
        self.image_path_label.setText(str(image_file[0]))

        # Removing the OS file path leaving on the filename and extension.
        image_file = ntpath.basename(image_file[0])

        # Removes the file extensions.
        self.image_num = image_file[0:-4]

        # Clearing existing text if any from the text browsers.
        self.actual_age_textBrowser.clear()
        self.gender_textBrowser.clear()
        self.predicted_age_textBrowser.clear()

    def estimate_boneage(self):
        # Checks if the image_num has been set if so the function continues if not a alert is shown.
        if self.image_num is 0:
            alert = QtWidgets.QMessageBox()
            alert.setWindowTitle("Image Selection Error!")
            alert.setText("Please select an image!")
            alert.setIcon(QtWidgets.QMessageBox.Critical)
            alert.exec_()
        else:

            input_image = int(self.image_num)

            # Disabling the predict button while the the images are being processed.
            self.predict_button.setEnabled(False)

            model_path = 'model/model.pth'
            images_path = 'data/eval_images/'

            img_name = os.path.join(images_path, str(input_image) + '.png')

            # Calculate mean and standard deviation to convert z-score back into months
            train_df = pd.read_csv('data/eval_data.csv', index_col='id')

            mean_boneage = train_df['boneage'].mean()
            std_boneage = train_df['boneage'].std()

            # Extract image data from evaluation set
            img_data = train_df.loc[input_image]

            # Declaring preprocessing transforms
            preprocess_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

            image = mpimg.imread(img_name)  # Read image
            image = Image.fromarray(image * 255).convert('RGB')  # Convert to RGB
            image = ImageOps.equalize(image, mask=None)  # Equalise image histogram
            image = preprocess_transform(image)  # Apply transforms
            image = image.unsqueeze(0)  # Reformat image

            # Use GPU for processing if available
            device = (torch.device("cuda") if torch.cuda.is_available() else "cpu")

            my_model = torch.load(model_path)  # Load model

            # Move variables to GPU
            image = image.to(device)
            age = torch.tensor(img_data['bone_age_z']).type(torch.float32).to(device)
            gender = torch.tensor(img_data['gender']).unsqueeze(0).to(device)

            with torch.no_grad():
                estimated_age = my_model(image, gender)  # Perform inference

            # Formatting resultant values in order to be shown in the appropriate TextBrowsers.
            actual_age = int((std_boneage * age.cpu().detach().item() + mean_boneage))
            if gender.item() == 1:
                actual_gender = "Male"
            else:
                actual_gender = "Female"
            predicted_age = int((std_boneage * estimated_age.cpu().detach().item() + mean_boneage))

            # Setting the Text Browsers values to the formatted results.
            self.actual_age_textBrowser.setText(str(actual_age) + " Months")
            self.gender_textBrowser.setText(str(actual_gender))
            self.predicted_age_textBrowser.setText(str(predicted_age) + " Months")

            # Re-enabling the predict button.
            self.predict_button.setEnabled(True)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())

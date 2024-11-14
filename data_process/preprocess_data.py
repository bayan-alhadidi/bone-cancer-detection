import cv2
import numpy as np
import os

train = "dataset/train/"
test = "dataset/test/"
validation = "dataset/validation/"

fTrain = "pre/train/"
fTest = "pre/test/"
fValidation = "pre/validation/"

classes = ['malignant', 'benign']

# create new folders for train, test, validation 
for directory in [fTrain, fTest, fValidation]:
    for class_name in classes:
        os.makedirs(os.path.join(directory, class_name), exist_ok=True)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

# Loop through all image files in the directory
for folder in [train, test, validation]:
    for class_name in classes:
        image_dir = os.path.join(folder, class_name)

        for filename in os.listdir(image_dir):
            file_path = os.path.join(image_dir, filename)
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                # Read the image
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

                # Apply Median filter
                filtered_image = cv2.medianBlur(image, 3)

                # contrast enhancement
                
                clahe_image = clahe.apply(filtered_image)

                # image sharpening
                
                sharpened_image = cv2.filter2D(clahe_image, -1, kernel)
                

                img_resized = cv2.resize(sharpened_image, (512, 512))

                # Save the filtered image
                if folder == train:
                    output_dir = os.path.join(fTrain, class_name)
                elif folder == test:
                    output_dir = os.path.join(fTest, class_name)
                else:
                    output_dir = os.path.join(fValidation, class_name)
                
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, img_resized)
    
print("images have been processed successfully")

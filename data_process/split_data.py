import os
import shutil
from sklearn.model_selection import train_test_split

original_data_dir = 'original_dataset/'
train = "dataset/train/"
test = "dataset/test/"
validation = "dataset/validation/"

classes = ['malignant', 'benign']

# create new folders for train, test, validation data
for directory in [train, test, validation]:
    for class_name in classes:
        os.makedirs(os.path.join(directory, class_name), exist_ok=True)

malignant_dir = os.path.join(original_data_dir, 'malignant')
benign_dir = os.path.join(original_data_dir, 'benign')
    
malignant_images = os.listdir(malignant_dir)
benign_images = os.listdir(benign_dir)

# split data for each class
def split_data(images):
    
    # split images to (train)(80%) and (test + validation)(20%)
    train_images, test_val_images = train_test_split(images, test_size=0.2, random_state=29)
    
    # split (test + validation) images to train (10%) and validation(10%)
    test_images, val_images = train_test_split(test_val_images, test_size=0.5, random_state=29)
    
    return train_images, val_images, test_images

# copy images to the new folders
for class_name, images in zip(classes, [malignant_images, benign_images]):
    train_images, val_images, test_images = split_data(images)
    
    for img_name in train_images:
        shutil.copy(os.path.join(original_data_dir, class_name, img_name), os.path.join(train, class_name))
        
    for img_name in val_images:
        shutil.copy(os.path.join(original_data_dir, class_name, img_name), os.path.join(validation, class_name))
        
    for img_name in test_images:
        shutil.copy(os.path.join(original_data_dir, class_name, img_name), os.path.join(test, class_name))
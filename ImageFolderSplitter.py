import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, Compose
from PIL import Image
from sklearn.model_selection import train_test_split

class ImageFolderSplitter:
    # images should be placed in folders like:
    # --root
    # ----root\dogs
    # ----root\dogs\image1.png
    # ----root\dogs\image2.png
    # ----root\cats
    # ----root\cats\image1.png
    # ----root\cats\image2.png    
    # path: the root of the image folder
    def __init__(self, path, train_size = 0.8):
        self.path = path
        self.train_size = train_size
        self.class2num = {}
        self.num2class = {}
        self.class_nums = {}
        self.data_x_path = []
        self.data_y_label = []
        self.x_train = []
        self.x_valid = []
        self.y_train = []
        self.y_valid = []
        for root, dirs, files in os.walk(path):
            if len(files) == 0 and len(dirs) > 1:
                for i, dir1 in enumerate(dirs):
                    self.num2class[i] = dir1
                    self.class2num[dir1] = i
            elif len(files) > 1 and len(dirs) == 0:
                category = ""
                for key in self.class2num.keys():
                    if key in root:
                        category = key
                        break
                label = self.class2num[category]
                self.class_nums[label] = 0
                for file1 in files:
                    self.data_x_path.append(os.path.join(root, file1))
                    self.data_y_label.append(label)
                    self.class_nums[label] += 1
            else:
                raise RuntimeError("please check the folder structure!")
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.data_x_path, self.data_y_label, shuffle = True, train_size = self.train_size)

    def getTrainingDataset(self):
        return self.x_train, self.y_train

    def getValidationDataset(self):
        return self.x_valid, self.y_valid

class DatasetFromFilename(Dataset):
    # x: a list of image file full path
    # y: a list of image categories
    def __init__(self, x, y, transforms = None):
        super(DatasetFromFilename, self).__init__()
        self.x = x
        self.y = y
        if transforms == None:
            self.transforms = ToTensor()
        else:
            self.transforms = transforms
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = Image.open(self.x[idx])
        img = img.convert("RGB")
        return self.transforms(img), torch.tensor([[self.y[idx]]])

# test code
# splitter = ImageFolderSplitter("for_test")
# transforms = Compose([Resize((51, 51)), ToTensor()])
# x_train, y_train = splitter.getTrainingDataset()
# training_dataset = DatasetFromFilename(x_train, y_train, transforms=transforms)
# training_dataloader = DataLoader(training_dataset, batch_size=2, shuffle=True)
# x_valid, y_valid = splitter.getValidationDataset()
# validation_dataset = DatasetFromFilename(x_valid, y_valid, transforms=transforms)
# validation_dataloader = DataLoader(validation_dataset, batch_size=2, shuffle=True)
# for x, y in training_dataloader:
#     print(x.shape, y.shape)
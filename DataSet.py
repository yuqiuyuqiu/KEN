import torch
import cv2
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, imList, labelList, transform=None):
        self.imList = imList
        self.labelList = labelList
        self.transform = transform

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, idx):
        image_name = self.imList[idx]
        label_name = self.labelList[idx]
        image = cv2.imread(image_name)
        image = image[:, :, ::-1]
        label = cv2.imread(label_name, 0)
        #label = label / 255
        if self.transform:
            [image, label] = self.transform(image, label)
        return (image, label)

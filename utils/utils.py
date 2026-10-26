import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
from torchvision import transforms


class Image_Datasets(Dataset):
    def __init__(self,root_dir,transform=None):
        self.transforms=transform
        self.root_dir=root_dir

        self.img_path=[os.path.join(root_dir,img) for img in os.listdir(root_dir)]
        self.img_path=[i for i in self.img_path if i.endswith(('jpg','png','jpeg'))]


    def __getitem__(self,idx):
        img_path=self.img_path[idx] 
        image=Image.open(img_path).convert('RGB')

        if self.transforms:
           image=self.transforms(image)
        return image

    def __len__(self):
        return len(self.img_path)

        
def get_transforms(img_size,crop,final_size):
    transform_list=[]
    if img_size>0:
       transform_list.append(transforms.Resize(img_size))
    if crop:
       transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize(final_size))

    transform_list.append(transforms.ToTensor())
    transformation=transforms.Compose(transform_list)

    return transformation    
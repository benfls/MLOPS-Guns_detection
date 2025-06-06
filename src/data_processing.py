import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class GunDataset(Dataset):

    def __init__(self, root:str , device:str = "cpu"):
        self.image_path = os.path.join(root, "Images")
        self.label_path = os.path.join(root, "Labels/")


        self.img_name = sorted(os.listdir(self.image_path))
        self.label_name = sorted(os.listdir(self.label_path))

        self.device = device

        logger.info("Data processing Initialized ....")

    def __getitem__(self, idx):
        try:

            logger.info(f"Loading data for index {idx}")

            image_path = os.path.join(self.image_path, str(self.img_name[idx]))
            logger.info(f"Image path : {image_path}")

            image = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            
            img_res = img_rgb / 255.0 # normalize the image
            img_res = torch.as_tensor(img_res).permute(2, 0, 1) # convert to tensor and change the order of dimensions

            #### Loading labels .....

            label_name = self.img_name[idx].rsplit('.', 1)[0] + ".txt"
            label_path = os.path.join(self.label_path, str(label_name))

            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file not found : {label_path}")
            
            target = {
                "boxes" : torch.tensor([]),
                "area" : torch.tensor([]),
                "image_id" : torch.tensor(idx),
                "labels" : torch.tensor([], dtype=torch.int64)
            }

            #print("Label path:", label_path)

            with open(label_path, "r") as label_file:
                l_count = int(label_file.readline())
                box = [list(map(int, label_file.readline().strip().split())) for _ in range(l_count)]

            if box:
                area = [(b[2] - b[0]) * (b[3] - b[1]) for b in box]
                labels = [1] * len(box)  # Assuming all boxes are of the same class (1 for gun)

                target["boxes"] = torch.tensor(box, dtype=torch.float32)
                target["area"] = torch.tensor(area, dtype=torch.float32)
                target["labels"] = torch.tensor(labels, dtype=torch.int64)

            img_res = img_res.to(self.device)
            for key in target:
                target[key] = target[key].to(self.device)

            return img_res, target
        
        except Exception as e:
            logger.error(f"Error loading data for index {idx}: {e}")
            raise CustomException("Failed to load data", e)
    
    def __len__(self):
        return len(self.img_name)
    
if __name__ == "__main__":
    root_path = "artifacts/raw"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = GunDataset(root_path, device)
    image, target = dataset[0]

    print("image shape : ", image.shape)
    print("target keys : ", target.keys())
    print("Bounding boxes : ", target["boxes"])
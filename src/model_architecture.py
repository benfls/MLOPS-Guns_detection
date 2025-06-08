import torch
from torch.optim import Adam
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from src.custom_exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)

class FasterRCNNModel:

    def __init__(self, num_classes, device):

        self.num_classes = num_classes
        self.device = device
        self.model = self.create_model().to(self.device)
        self.optimizer = None
        logger.info("Model architecture initialized.")

    def create_model(self):
        try:

            model = fasterrcnn_resnet50_fpn(pretrained=True)
            in_features = model.roi_heads.box_predictor.cls_score.in_features

            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

            return model
        except Exception as e:
            logger.error(f"Error in creating model: {e}")
            raise CustomException(f"Error in creating model: {e}")
        
    def compile(self, lr=1e-4):
        try:
            self.optimizer = Adam(self.model.parameters(), lr=lr)
            logger.info("Model compiled with optimizer.")
        except Exception as e:
            logger.error(f"Error in compiling model: {e}")
            raise CustomException(f"Error in compiling model: {e}")
    
    def train(self, train_loader, num_epochs=10):
        try:
            self.model.train()
            for epoch in range(num_epochs):
                total_loss = 0
                logger.info(f"Epoch {epoch} started .....")
                for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}"):
                    images = [img.to(self.device) for img in images]
                    targets = [{key: value.to(self.device) for key, value in target.items()}for target in targets]

                    loss_dict = self.model(images, targets)
                    loss = sum(loss for loss in loss_dict.values())

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()

                logger.info(f"Epoch {epoch} completed with total loss: {total_loss}")
        except Exception as e:
            logger.error(f"Error in training model: {e}")
            raise CustomException(f"Error in training model: {e}")
        
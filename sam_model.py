from segment_anything import sam_model_registry, SamPredictor
import torch
import numpy as np
from PIL import Image

class SAMModel:
    def __init__(self, sam_checkpoint="checkpoints/sam_vit_h_4b8939.pth", model_type="vit_h"):
        """
        Initializes the SAMModel.

        Args:
            sam_checkpoint (str): Path to the SAM model checkpoint.
            model_type (str): Type of the SAM model (e.g., "vit_h").
        """
        self.sam_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SAM model from {sam_checkpoint} for device: {self.sam_device}")
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=self.sam_device)
        self.predictor = SamPredictor(self.sam)
        print("SAM model loaded successfully.")

    def predict_mask(self, image: Image.Image, input_box: np.ndarray):
        """
        Predicts a segmentation mask for a given image and bounding box.

        Args:
            image (PIL.Image.Image): The input image.
            input_box (np.ndarray): Bounding box coordinates in [x1, y1, x2, y2] format.

        Returns:
            np.ndarray: The predicted binary mask.
        """
        img_np = np.array(image)
        self.predictor.set_image(img_np)
        masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=False,
        )
        return masks[0]
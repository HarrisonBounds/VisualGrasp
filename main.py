from PIL import Image, ImageDraw
import numpy as np
from sam_model import SAMModel
from gemini_model import GeminiModel
import os

def main():
    # Initialize models
    sam_model = SAMModel()
    gemini_model = GeminiModel()

    img_path = "imgs/demo.jpeg"
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return

    image = Image.open(img_path).convert("RGB")
    original_width, original_height = image.size

    # Prompt for Gemini Model
    object_name = "mustard bottle"
    parsed_data = gemini_model.get_bounding_box(image, object_name)

    if object_name in parsed_data:
        coordinate_list = parsed_data[object_name]

        ymin_norm, xmin_norm, ymax_norm, xmax_norm = coordinate_list

        scale_factor = 1000.0 # Gemini's normalized coordinates are typically out of 1000

        x1 = int(xmin_norm / scale_factor * original_width)
        y1 = int(ymin_norm / scale_factor * original_height)
        x2 = int(xmax_norm / scale_factor * original_width)
        y2 = int(ymax_norm / scale_factor * original_height)

        input_box_for_sam = np.array([x1, y1, x2, y2])

        # Debug: Save image with Gemini's bounding box
        temp_image_with_box = image.copy()
        draw = ImageDraw.Draw(temp_image_with_box)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        temp_image_with_box.save(f"imgs/{object_name}_bbox.png")
        print("Saved debug_gemini_bbox.png with Gemini's predicted bounding box.")

        # Predict mask using SAM
        mask = sam_model.predict_mask(image, input_box_for_sam)

        # Apply mask to image
        img_np = np.array(image)
        red_layer = (mask[..., None] * np.array([0, 0, 255], dtype=np.uint8))
        blended = ((img_np.astype(float) * 0.5) + (red_layer.astype(float) * 0.5)).astype(np.uint8)

        masked_image = Image.fromarray(blended)
        output_filename = f"imgs/{object_name}_mask.png"
        masked_image.save(output_filename)
        print(f"Segmentation output saved to {output_filename}")
    else:
        print(f"Object '{object_name}' not found in Gemini's response.")

if __name__ == "__main__":
    main()
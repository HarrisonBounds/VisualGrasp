from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from segment_anything import sam_model_registry, SamPredictor
import torch
from PIL import Image, ImageDraw
import re
import json
import numpy as np
import google.generativeai as genai 
import os 

def main():
    original_width, original_height = 0, 0

    #SAM model
    sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h" 
    sam_device = "cuda" if torch.cuda.is_available() else "cpu" 
    
    #Gemini Model
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')

    img_path = "imgs/demo.jpeg"
    image = Image.open(img_path).convert("RGB")
    original_width, original_height = image.size

    #Load SAM Model
    print("Loading SAM")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=sam_device)
    sam_predictor = SamPredictor(sam)

    print(f"SAM Model is on device: {sam_device}")

    #Prompt for Qwen Model
    object_name = "vita lemon tea"
    prompt = f"""
                Analyze the following image and provide the bounding\
                box of the [{object_name}]. Bounding boxes should be\
                in the format [ymin, xmin, ymax, xmax]. Additional\
                notes:\n
                * Please ensure the coordinates are relative to\
                  the original image size.\n
                * If an object is partially out of frame, estimate\
                  the bounding box as best as possible.\n
                * Return your answer as a single dict object where\
                  each key is an object name and each value is the\
                  corresponding bounding box coordinates. For example,\
                  {object_name}: [ymin, xmin, ymax, xmax]. Do not\
                  use Markdown. where H and W are the original image\
                  size.\n
                * If there are mutiple same object, you can name\
                  the the second object as {object_name}_2.
                  """
    
    content = [image, prompt]

    #Process inputs
    response = gemini_model.generate_content(content)
    response_text = response.text

    print(f"Response: {response_text}")

    json_string_match = re.search(r'\{.*\}', response_text, re.DOTALL)

    if json_string_match:
        json_string = json_string_match.group(0)        
        parsed_data = json.loads(json_string)

        if object_name in parsed_data:
            coordinate_list = parsed_data[object_name]

            print(f"Coordinates: {coordinate_list}")

            ymin_norm, xmin_norm, ymax_norm, xmax_norm = coordinate_list

            scale_factor = 1000.0
                
            x1 = int(xmin_norm / scale_factor * original_width)
            y1 = int(ymin_norm / scale_factor * original_height)
            x2 = int(xmax_norm / scale_factor * original_width)
            y2 = int(ymax_norm / scale_factor * original_height)
            

            input_box_for_sam = np.array([x1, y1, x2, y2])

            temp_image_with_box = image.copy()
            draw = ImageDraw.Draw(temp_image_with_box)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            temp_image_with_box.save("debug_gemini_bbox.png")

            image_np = np.array(image)
            sam_predictor.set_image(image_np)

            masks, scores, logits = sam_predictor.predict(
                    point_coords=None, 
                    point_labels=None, 
                    box=input_box_for_sam, 
                    multimask_output=False, 
                )

            mask = masks[0]

            masked_image_output = np.zeros_like(image_np)
            masked_image_output[mask] = image_np[mask]
            final_segmented_image = Image.fromarray(masked_image_output)

            output_filename = "segmented_output_gemini.png"
            final_segmented_image.save(output_filename)

main()
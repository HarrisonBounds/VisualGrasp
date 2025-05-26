from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch
from PIL import Image

def main():
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    img_path = "imgs/desk.jpeg"
    image = Image.open(img_path).convert("RGB")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(model_id)

    print(f"Model is on device: {model.device}")

    object_name = "coffee can"
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
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": prompt},  
            ],
        }
    ]

    inputs = processor(
        text=processor.apply_chat_template(conversation, add_generation_prompt=True),
        images=[image],
        return_tensors="pt" 
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    #Inference
    with torch.no_grad(): 
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=256, 
            do_sample=True,     
            temperature=0.7,    
            top_p=0.9           
        )

    response_text = processor.decode(output_tokens[0], skip_special_tokens=True)
    print("\nModel's raw response:")
    print(response_text)

main()
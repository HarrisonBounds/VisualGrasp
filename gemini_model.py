import google.generativeai as genai
import os
import re
import json
from PIL import Image

class GeminiModel:
    def __init__(self):
        """
        Initializes the GeminiModel.
        Requires GOOGLE_API_KEY to be set in environment variables.
        """
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        print("Gemini model initialized.")

    def get_bounding_box(self, image: Image.Image, object_name: str) -> dict:
        """
        Uses the Gemini model to get the bounding box of an object in an image.

        Args:
            image (PIL.Image.Image): The input image.
            object_name (str): The name of the object to find.

        Returns:
            dict: A dictionary containing the parsed bounding box data,
                  e.g., {"object_name": [ymin, xmin, ymax, xmax]}.
                  Returns an empty dictionary if no valid JSON is found.
        """
        prompt = f"""
                Analyze the following image and provide the bounding\\
                box of the [{object_name}]. Bounding boxes should be\\
                in the format [ymin, xmin, ymax, xmax]. Additional\\
                notes:\\n
                * Please ensure the coordinates are relative to\\
                  the original image size.\\n
                * If an object is partially out of frame, estimate\\
                  the bounding box as best as possible.\\n
                * Return your answer as a single JSON object where\\
                  each key is an object name and each value is the\\
                  corresponding bounding box coordinates. For example,\\
                  {{\"{object_name}\": [ymin, xmin, ymax, xmax]}}. Do not\\
                  use Markdown. where H and W are the original image\\
                  size.\\n
                * If there are mutiple same object, you can name\\
                  the the second object as {object_name}_2.
                * Make sure the coordinates encapsulate the entire object, \\
                  a bigger box is better than a smaller one
                  """
        content = [image, prompt]
        response = self.model.generate_content(content)
        response_text = response.text
        print(f"Gemini Response: {response_text}")

        json_string_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_string_match:
            json_string = json_string_match.group(0)
            return json.loads(json_string)
        return {}
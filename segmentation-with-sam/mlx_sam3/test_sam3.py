from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()  # first run downloads weights, will take a minute
processor = Sam3Processor(model, confidence_threshold=0.5)

image = Image.open("IMG_1.jpg")  # replace with your actual filename
state = processor.set_image(image)
state = processor.set_text_prompt("sidewalk", state)  # or whatever object you want

masks = state["masks"]
boxes = state["boxes"]
scores = state["scores"]

print(f"Found {len(scores)} objects")
print(f"Scores: {scores}")

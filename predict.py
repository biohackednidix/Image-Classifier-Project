import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import argparse

# Define command-line arguments
parser = argparse.ArgumentParser(description="Predict image class")
parser.add_argument("image_path", type=str, help="Path to image")
parser.add_argument("checkpoint", type=str, help="Path to trained model checkpoint")
parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes")
parser.add_argument("--category_names", type=str, default=None, help="Path to category names JSON file")
parser.add_argument("--gpu", action="store_true", help="Use GPU if available")

args = parser.parse_args()

# Load checkpoint
checkpoint = torch.load(args.checkpoint, map_location="cuda" if args.gpu and torch.cuda.is_available() else "cpu")
model_arch = checkpoint["arch"]

if model_arch == "vgg16":
    model = models.vgg16(pretrained=True)
    
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Define image processing function
def process_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Predict function
def predict(image_path, model, topk=5):
    model.to("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    image = process_image(image_path)
    image = image.to("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.exp(output)
    
    top_probs, top_classes = probabilities.topk(topk, dim=1)
    return top_probs.squeeze().tolist(), top_classes.squeeze().tolist()

# Get predictions
probs, classes = predict(args.image_path, model, args.top_k)

# Load category names
if args.category_names:
    with open(args.category_names, "r") as f:
        cat_to_name = json.load(f)
    classes = [cat_to_name[str(cls)] for cls in classes]

print("Predicted classes and probabilities:")
for i in range(len(probs)):
    print(f"{classes[i]}: {probs[i]:.3f}")

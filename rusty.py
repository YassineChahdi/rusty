import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from torch.nn.functional import softmax


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 2),
    )
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()
    
    return model

def predict_image(path, model_path="models/rusty55_small.pth", threshold=0.55):
    """
    Given an image of a car, returns wether car contains rust or not.

    :param path: File path to the image.
    :param model: The model.
    :param threshold: Threshold at which decision is made.
    :param eval_tf: The evaluation transforms.
    :return: 'rust' if the car contains rust, 'clean' otherwise, followed by the rust probability.
    """
    model = load_model(model_path)

    tf = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(288),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    img = Image.open(path).convert("RGB")
    x = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p_rust = softmax(model(x), dim=1)[0, 1].item()
    label = "rust" if p_rust >= threshold else "clean"
    return label, p_rust

from utils import estimate_training_time, load_model, load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
import streamlit as st

data_loader, class_map = load_dataset("data3/data3_test", batch_size=64,split=False, return_paths=True)
model = load_model(num_classes=len(class_map))
model.load_state_dict(torch.load("best_model_fc_layer3_layer4.pth"))
model.eval()

class_names = data_loader.dataset.classes

image_size = 224

transform = T.Compose([
    T.Resize((image_size, image_size)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

st.title("Furniture Classifier")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width =True)
    x = transform(img).unsqueeze(0).to(device)  # ✅ send to same device
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).squeeze()
        topk = torch.topk(probs, 5)

    st.write("### Top 5 Predictions:")
    for idx, score in zip(topk.indices, topk.values):
        st.write(f"{class_names[idx]}: {score.item()*100:.1f}%")
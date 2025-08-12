from utils import estimate_training_time, load_model, load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
import streamlit as st
import gdown
import os

class_map = {
 'DRZWI, OKNA, DEKORACJE': 0,
 'ELEKTRONIKA, INNE URZĄDZENIA': 1,
 'ELEMENTY DEKORACYJNE': 2,
 'GRZEJNIKI': 3,
 'INNE ELEMENTY WYPOSAŻENIA WNĘTRZ': 4,
 'KOMINKI I AKCESORIA': 5,
 'KOMODY, KONSOLE, TOALETKI': 6,
 'KRZESŁA, HOKERY, TABORETY': 7,
 'OGRÓD': 8,
 'OŚWIETLENIE': 9,
 'REGAŁY, PÓŁKI, WITRYNY': 10,
 'SIEDZISKA': 11,
 'SPORT I ROZRYWKA': 12,
 'STOŁY, STOLIKI': 13,
 'SYMBOLE ARCHITEKTONICZNO-BUDOWLANE': 14,
 'SZAFY, SZAFKI': 15,
 'TEKSTYLIA': 16,
 'WYPOSAŻENIE BIURA': 17,
 'WYPOSAŻENIE KUCHNI': 18,
 'WYPOSAŻENIE POKOJU DZIECIĘCEGO': 19,
 'WYPOSAŻENIE ŁAZIENEK': 20,
 'ŁÓŻKA': 21
}

model = load_model(num_classes=len(class_map))

MODEL_PATH = "best_model_fc_layer3_layer4.pth"

# if not os.path.exists(MODEL_PATH):
file_id = "1YXb14nU9AbGV5o3mUM0W1bOsSOOkos0h"
url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, MODEL_PATH, quiet=False)

model.load_state_dict(torch.load("best_model_fc_layer3_layer4.pth"))
model.eval()

class_names = ['DRZWI, OKNA, DEKORACJE', 'ELEKTRONIKA, INNE URZĄDZENIA', 'ELEMENTY DEKORACYJNE', 'GRZEJNIKI', 'INNE ELEMENTY WYPOSAŻENIA WNĘTRZ', 'KOMINKI I AKCESORIA', 'KOMODY, KONSOLE, TOALETKI', 'KRZESŁA, HOKERY, TABORETY', 'OGRÓD', 'OŚWIETLENIE', 'REGAŁY, PÓŁKI, WITRYNY', 'SIEDZISKA', 'SPORT I ROZRYWKA', 'STOŁY, STOLIKI', 'SYMBOLE ARCHITEKTONICZNO-BUDOWLANE', 'SZAFY, SZAFKI', 'TEKSTYLIA', 'WYPOSAŻENIE BIURA', 'WYPOSAŻENIE KUCHNI', 'WYPOSAŻENIE POKOJU DZIECIĘCEGO', 'WYPOSAŻENIE ŁAZIENEK', 'ŁÓŻKA']

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
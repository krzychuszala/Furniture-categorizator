import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import streamlit as st
import gdown
import os
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights

# === NEW: Google Sheets imports ===
import gspread
from google.oauth2.service_account import Credentials

scope = ["https://www.googleapis.com/auth/spreadsheets",
         "https://www.googleapis.com/auth/drive"]

creds = Credentials.from_service_account_file("streamlite_app/service_account.json", scopes=scope)
client = gspread.authorize(creds)

sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1xeW01_Xr9-IS3yfGkVKy1gaX6bxpGUG58h5Uq9XvWYk/edit").sheet1

# ======================
# MODEL + CLASSES
# ======================

MODEL_PATH = "efficientnet_v2_s_fc_layer3_layer4.pth"

if not os.path.exists(MODEL_PATH):
    file_id = "1j9HjVa8jhzV71XPEonBgr5RVqbcU39nQ"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)

def load_efficientnet_v2_s(num_classes, weights_path=None):
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

class_names = [
 'DRZWI, OKNA, DEKORACJE','ELEKTRONIKA, INNE URZĄDZENIA','ELEMENTY DEKORACYJNE',
 'GRZEJNIKI','INNE ELEMENTY WYPOSAŻENIA WNĘTRZ','KOMINKI I AKCESORIA',
 'KOMODY, KONSOLE, TOALETKI','KRZESŁA, HOKERY, TABORETY','OGRÓD','OŚWIETLENIE',
 'REGAŁY, PÓŁKI, WITRYNY','SIEDZISKA','SPORT I ROZRYWKA','STOŁY, STOLIKI',
 'SYMBOLE ARCHITEKTONICZNO-BUDOWLANE','SZAFY, SZAFKI','TEKSTYLIA',
 'WYPOSAŻENIE BIURA','WYPOSAŻENIE KUCHNI','WYPOSAŻENIE POKOJU DZIECIĘCEGO',
 'WYPOSAŻENIE ŁAZIENEK','ŁÓŻKA'
]

model = load_efficientnet_v2_s(num_classes=len(class_names),
                               weights_path="efficientnet_v2_s_fc_layer3_layer4.pth")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ======================
# STREAMLIT UI
# ======================
st.sidebar.title("Categories")
for c in class_names:
    st.sidebar.write(c)

st.title("Furniture Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).squeeze()
        topk = torch.topk(probs, 5)

    # Show predictions
    st.write("### Top 5 Predictions:")
    pred_list = []
    for idx, score in zip(topk.indices, topk.values):
        line = f"{class_names[idx]}: {score.item()*100:.1f}%"
        st.write(line)
        pred_list.append(line)

    # User selects true class
    true_label = st.selectbox("Select the correct class:", class_names)

    if st.button("✅ Save to Google Sheet"):
        filename = uploaded_file.name
        top_pred = class_names[topk.indices[0]]
        is_correct = (top_pred == true_label)

        # Append row to your Google Sheet
        sheet.append_row([
            filename,
            true_label,
            top_pred,
            str(is_correct),
            "; ".join(pred_list)
        ])

        st.success("✅ Saved result to Google Sheet!")


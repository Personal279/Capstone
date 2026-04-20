#
#import os
#import cv2
#import numpy as np
#import pandas as pd
#from PIL import Image
#
#import torch
#from torchvision.models import resnet50, ResNet50_Weights
#import torchvision.transforms as transforms
#
#import mediapipe as mp
#
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.neural_network import MLPClassifier
#from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#from sklearn.utils.class_weight import compute_sample_weight  # ← NEW
#import joblib
#
## =========================
## CONFIG
## =========================
#CSV_PATH = r"C:\Users\jagadeessh\Capstone\labels_fixed.csv"
#SAVE_DIR = r"C:\Users\jagadeessh\Capstone\saved_models"
#os.makedirs(SAVE_DIR, exist_ok=True)
#
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"🖥️ Using device: {device}")
#
## =========================
## MEDIAPIPE
## =========================
#mp_face   = mp.solutions.face_mesh
#face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
#
#def extract_fai(image_path):
#    img = cv2.imread(image_path)
#    if img is None:
#        return np.zeros(1)
#    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    result  = face_mesh.process(img_rgb)
#    if not result.multi_face_landmarks:
#        return np.zeros(1)
#    lm   = result.multi_face_landmarks[0].landmark
#    eye  = abs(lm[33].y  - lm[263].y)
#    lip  = abs(lm[61].y  - lm[291].y)
#    brow = abs(lm[70].y  - lm[300].y)
#    cheek= abs(lm[50].y  - lm[280].y)
#    jaw  = abs(lm[152].y - lm[10].y)
#    fai  = (0.30*eye + 0.30*lip + 0.15*brow + 0.15*cheek + 0.10*jaw)
#    return np.array([fai])
#
## =========================
## RESNET50
## =========================
#weights = ResNet50_Weights.DEFAULT
#resnet  = resnet50(weights=weights)
#resnet  = torch.nn.Sequential(*list(resnet.children())[:-1])
#resnet  = resnet.to(device)
#resnet.eval()
#
#_mean = weights.transforms().mean
#_std  = weights.transforms().std
#
#transform = transforms.Compose([
#    transforms.Resize((224, 224)),
#    transforms.ToTensor(),
#    transforms.Normalize(mean=_mean, std=_std)
#])
#
#def extract_resnet(image_path):
#    try:
#        img = Image.open(image_path).convert("RGB")
#    except:
#        return np.zeros(2048)
#    tensor = transform(img).unsqueeze(0).to(device)
#    with torch.no_grad():
#        feat = resnet(tensor)
#    return feat.cpu().flatten().numpy()
#
#def extract_features(path):
#    fai = extract_fai(path) * 10   # same as original ✅
#    res = extract_resnet(path)
#    return np.concatenate([fai, res])
#
## =========================
## LOAD CSV
## =========================
#print("\n📂 Loading CSV...")
#df = pd.read_csv(CSV_PATH)
#df.columns = df.columns.str.strip().str.lower()
#print("Rows:", len(df))
#
#X, y = [], []
#print("🔄 Extracting features...")
#
#for i, row in df.iterrows():
#    path  = row["image_path"]
#    label = int(row["grade"])
#    if not os.path.exists(path):
#        continue
#    feat = extract_features(path)
#    X.append(feat)
#    y.append(label)
#    if (i+1) % 100 == 0:
#        print(f"✔ {i+1}/{len(df)} processed")
#
#X = np.array(X)
#y = np.array(y)
#print("✅ Features ready:", X.shape)
#
## =========================
## NORMALIZATION
## =========================
#scaler = StandardScaler()
#X = scaler.fit_transform(X)
#joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))
#
## =========================
## SPLIT
## =========================
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.2, random_state=42, stratify=y
#)
#
## ← NEW: class weights to handle imbalance
#sample_weights = compute_sample_weight("balanced", y_train)
#
## =========================
## MLP — exact same as original
## =========================
#model = MLPClassifier(
#    hidden_layer_sizes=(1024, 512, 256),
#    activation='relu',
#    learning_rate_init=0.0005,
#    max_iter=400,
#    early_stopping=True,
#    random_state=42,
#    verbose=True
#)
#
#print("🚀 Training...")
#model.fit(X_train, y_train, sample_weight=sample_weights)  # ← NEW
#
#joblib.dump(model, os.path.join(SAVE_DIR, "mlp.pkl"))
#
## =========================
## EVALUATION
## =========================
#y_pred = model.predict(X_test)
#acc    = accuracy_score(y_test, y_pred)
#
#print("\n🎯 Accuracy:", acc*100, "%")
#print("\n📊 Report:\n", classification_report(y_test, y_pred))
#print("\n📊 Confusion:\n", confusion_matrix(y_test, y_pred))
#
#face_mesh.close()
#print("\n✅ DONE")
#



import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms

import mediapipe as mp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import joblib

# =========================
# CONFIG
# =========================
CSV_PATH = r"C:\Users\jagadeessh\Capstone\balanced_labels_1.csv"
SAVE_DIR = r"C:\Users\jagadeessh\Capstone\saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# =========================
# MEDIAPIPE
# =========================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def extract_fai(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros(1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(img_rgb)
    if not result.multi_face_landmarks:
        return np.zeros(1)
    lm = result.multi_face_landmarks[0].landmark
    eye  = abs(lm[33].y  - lm[263].y)
    lip  = abs(lm[61].y  - lm[291].y)
    brow = abs(lm[70].y  - lm[300].y)
    cheek= abs(lm[50].y  - lm[280].y)
    jaw  = abs(lm[152].y - lm[10].y)
    fai  = (0.30*eye + 0.30*lip + 0.15*brow + 0.15*cheek + 0.10*jaw)
    return np.array([fai])

# =========================
# RESNET50
# =========================
weights = ResNet50_Weights.DEFAULT
resnet = resnet50(weights=weights)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet = resnet.to(device)
resnet.eval()

_mean = weights.transforms().mean
_std  = weights.transforms().std

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_mean, std=_std)
])

def extract_resnet(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
    except:
        return np.zeros(2048)
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = resnet(tensor)
    return feat.cpu().flatten().numpy()

def extract_features(path):
    fai = extract_fai(path) * 10
    res = extract_resnet(path)
    return np.concatenate([fai, res])

# =========================
# LOAD CSV
# =========================
print("\n📂 Loading CSV...")
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip().str.lower()
print("Rows:", len(df))

X, y = [], []
print("🔄 Extracting features...")

for i, row in df.iterrows():
    path  = row["image_path"]
    label = int(row["grade"])
    if not os.path.exists(path):
        continue
    feat = extract_features(path)
    X.append(feat)
    y.append(label)
    if (i+1) % 100 == 0:
        print(f"✔ {i+1}/{len(df)} processed")

X = np.array(X)
y = np.array(y)
print("✅ Features ready:", X.shape)

# =========================
# SPLIT (🔥 BEFORE SCALING)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# NORMALIZATION (🔥 FIXED)
# =========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)  # fit only on train
X_test  = scaler.transform(X_test)       # apply to test

joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))

# =========================
# CLASS WEIGHTING
# =========================
sample_weights = compute_sample_weight("balanced", y_train)
sample_weights = np.array(sample_weights)

for i in range(len(y_train)):
    if y_train[i] == 5:
        sample_weights[i] *= 5
    elif y_train[i] == 4:
        sample_weights[i] *= 2

# =========================
# MLP MODEL
# =========================
model = MLPClassifier(
    hidden_layer_sizes=(1024, 512, 256),
    activation='relu',
    learning_rate_init=0.0005,
    max_iter=400,
    early_stopping=True,
    random_state=42,
    verbose=True
)

print("🚀 Training...")
model.fit(X_train, y_train, sample_weight=sample_weights)

joblib.dump(model, os.path.join(SAVE_DIR, "mlp.pkl"))

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n🎯 Accuracy:", acc * 100, "%")
print("\n📊 Report:\n", classification_report(y_test, y_pred))
print("\n📊 Confusion:\n", confusion_matrix(y_test, y_pred))

face_mesh.close()
print("\n✅ DONE")
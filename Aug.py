#import os
#import pandas as pd
#from PIL import Image
#import torchvision.transforms as transforms
#
## ---- CONFIG ----
#CSV_PATH = "labels.csv"
#
#OUTPUT_DIR = r"C:\Users\jagadeessh\Capstone\palsynet\palsynet_frames\frames\train\normal\aug_grade5"
#
#TARGET_COUNT = 250
#GRADE = 5
#
#os.makedirs(OUTPUT_DIR, exist_ok=True)
#
## ---- LOAD CSV ----
#df = pd.read_csv(CSV_PATH, header=None)
#
## 🔍 DEBUG: Check how grades are stored
#print("Unique grades in CSV:", df[1].unique())
#
## ---- FIX TYPE ISSUE ----
## Convert grade column to string and remove spaces
#df[1] = df[1].astype(str).str.strip()
#
## Filter Grade 5
#df_grade5 = df[df[1] == str(GRADE)]
#
#image_paths = df_grade5[0].tolist()
#
#print(f"Found {len(image_paths)} Grade {GRADE} images")
#
## ---- SAFETY CHECK ----
#if len(image_paths) == 0:
#    print("❌ No Grade 5 images found. Check CSV format.")
#    exit()
#
## ---- AUGMENTATION PIPELINE ----
#transform = transforms.Compose([
#    transforms.RandomHorizontalFlip(p=0.5),
#    transforms.RandomRotation(10),
#    transforms.ColorJitter(brightness=0.15, contrast=0.15),
#    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
#])
#
## ---- GENERATE AUGMENTED IMAGES ----
#new_rows = []
#count = 0
#
#while count < TARGET_COUNT:
#    for i, img_path in enumerate(image_paths):
#        if count >= TARGET_COUNT:
#            break
#        
#        try:
#            img = Image.open(img_path).convert("RGB")
#            aug_img = transform(img)
#
#            filename = f"aug_{count}.jpg"
#            save_path = os.path.join(OUTPUT_DIR, filename)
#
#            aug_img.save(save_path)
#
#            # Copy original row metadata
#            row = df_grade5.iloc[i].copy()
#
#            # Save relative path in CSV
#            row[0] = f"train/normal/aug_grade5/{filename}"
#
#            new_rows.append(row)
#
#            count += 1
#
#        except Exception as e:
#            print(f"Error with {img_path}: {e}")
#
#print(f"✅ Generated {count} augmented images")
#
## ---- APPEND TO CSV ----
#new_df = pd.DataFrame(new_rows)
#updated_df = pd.concat([df, new_df], ignore_index=True)
#
#updated_df.to_csv("labels_updated.csv", index=False, header=False)
#
#print("✅ CSV updated successfully!")


import pandas as pd

df = pd.read_csv("labels_balanced.csv", header=None)

# normalize column (important)
df[1] = df[1].astype(str).str.strip()

print(df[1].value_counts())

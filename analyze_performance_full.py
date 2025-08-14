import os
from ultralytics import YOLO
import pandas as pd

# =========================
# 1️⃣ Automatically find the latest weights
# =========================
runs_detect = r"F:\Downloads\LEGO-Detection\runs\detect"  # base detect folder
train_folders = [f for f in os.listdir(runs_detect) if f.startswith("train")]

if not train_folders:
    raise FileNotFoundError("No 'train*' folders found in runs/detect!")

# Sort by modified time to get the latest training run
train_folders.sort(key=lambda x: os.path.getmtime(os.path.join(runs_detect, x)), reverse=True)
latest_train = train_folders[0]

weights_path = os.path.join(runs_detect, latest_train, "weights", "best.pt")
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"No 'best.pt' found in {os.path.join(runs_detect, latest_train, 'weights')}")

print(f"Using weights: {weights_path}")

# =========================
# 2️⃣ Load the YOLO model
# =========================
model = YOLO(weights_path)

# =========================
# 3️⃣ Evaluate on validation dataset
# =========================
val_dataset = r"F:\Downloads\LEGO-Detection\lego_dataset\valid"  # path to validation set
results = model.val(data=val_dataset, save=True, save_txt=True)  # saves images + txt results

# =========================
# 4️⃣ Extract per-class metrics safely
# =========================
metrics_list = []

# results is a list of Metric objects (usually one element for a single validation dataset)
for metric in results:
    class_names = metric.names  # get class names from the metric
    for i, cls_name in enumerate(class_names):
        metrics_list.append({
            "Class": cls_name,
            "Precision": round(float(metric.p[i]), 3),
            "Recall": round(float(metric.r[i]), 3),
            "mAP50": round(float(metric.ap50[i]), 3),
            "mAP50-95": round(float(metric.ap[i]), 3)
        })

# Convert to DataFrame for readability
df_metrics = pd.DataFrame(metrics_list)
print("\n✅ Per-Class Metrics:")
print(df_metrics)

# Save to CSV for record
metrics_csv = os.path.join(runs_detect, latest_train, "metrics_per_class.csv")
df_metrics.to_csv(metrics_csv, index=False)
print(f"\nMetrics saved to: {metrics_csv}")
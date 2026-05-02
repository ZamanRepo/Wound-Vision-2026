# test_prediction.py

import pandas as pd
import numpy as np
from models.prediction import predict_healing_time
from explainability.xai_tools import generate_explanations

# Load CSV
csv_path = r'C:\Users\rudri\Downloads\ds\masked_wound_areas.csv'  # Update if the path is different
df = pd.read_csv(csv_path)

# Sample a row for testing
sample = df.iloc[0]
age = sample['age']
area = sample['area_pixels']  # This is in mm² already
diabetic_status = sample['diabetic_status']

# Prepare the metadata array (shape must be (1, 10) for compatibility)
metadata = np.zeros((1, 10))
metadata[0, 0] = age
metadata[0, 1] = diabetic_status
metadata[0, 9] = area / 100  # convert to cm² if you want to match UI logic

# Predict healing time
predicted_days = predict_healing_time(metadata)
explanation = generate_explanations(metadata)

# Output
print(f"🩺 Age: {age}")
print(f"🩹 Diabetic Status: {diabetic_status}")
print(f"📏 Wound Area: {area:.2f} mm²")
print(f"🕒 Predicted Healing Time: {predicted_days:.2f} days")
print(f"🔍 Explanation: {explanation}")

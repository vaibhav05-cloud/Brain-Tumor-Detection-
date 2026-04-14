import os
os.environ["TF_USE_LEGACY_KERAS"] = "0"

import keras

MODEL_PATH = r'E:\brain tumor detection\model\my_tumor_detection.keras'
FIXED_PATH = r'E:\brain tumor detection\model\my_tumor_detection_fixed.h5'

print("⏳ Loading...")
model = keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Loaded!")

model.save(FIXED_PATH)
print("✅ Saved as .h5!")
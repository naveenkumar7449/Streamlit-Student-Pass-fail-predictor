import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate student records
n_samples = 300  # Reduced dataset size for faster training
study_time = np.random.normal(10, 3, n_samples)  # Mean 10 hours, std 3 hours
attendance = np.random.normal(4, 1, n_samples)   # Mean 4 days, std 1 day

# Ensure values are realistic
study_time = np.clip(study_time, 0, 20)   # Limit study time between 0-20 hours
attendance = np.clip(attendance, 0, 5)    # Limit attendance between 0-5 days

# Calculate fail probability: Less study & less attendance → More likely to fail
fail_probability = -study_time / 8 - attendance / 3 + np.random.normal(0, 0.6, n_samples)

# Use median threshold to balance pass/fail classes
fail = (fail_probability > np.median(fail_probability)).astype(int)

# Create DataFrame
df = pd.DataFrame({
    "study_time": study_time,
    "attendance": attendance,
    "fail": fail
})

# Save dataset
df.to_csv("data.csv", index=False)
print("Dataset saved as data.csv")

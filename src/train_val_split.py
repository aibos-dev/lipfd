import os
import shutil

# Path to the original training data
train_real_path = "datasets/AVLips/0_real"
train_fake_path = "datasets/AVLips/1_fake"

# Path to the validation data
val_real_path = "datasets/val/0_real"
val_fake_path = "datasets/val/1_fake"

# Path to the new training data
new_train_real_path = "datasets/AVLips_new/0_real"
new_train_fake_path = "datasets/AVLips_new/1_fake"

# Ensure the new training data folders exist
os.makedirs(new_train_real_path, exist_ok=True)
os.makedirs(new_train_fake_path, exist_ok=True)

# Get the list of filenames from the validation data folders
val_real_files = set(os.listdir(val_real_path))
val_fake_files = set(os.listdir(val_fake_path))

# Print initial file count information
print("Initial file count information:")
print(f"Original training data folder - Real data: {len(os.listdir(train_real_path))} files")
print(f"Original training data folder - Fake data: {len(os.listdir(train_fake_path))} files")
print(f"Validation data folder - Real data: {len(val_real_files)} files")
print(f"Validation data folder - Fake data: {len(val_fake_files)} files")

# Process real data
for filename in os.listdir(train_real_path):
    if filename not in val_real_files:  # If the file is not in the validation set
        source_file = os.path.join(train_real_path, filename)
        dest_file = os.path.join(new_train_real_path, filename)
        shutil.copy(source_file, dest_file)

# Process fake data
for filename in os.listdir(train_fake_path):
    if filename not in val_fake_files:  # If the file is not in the validation set
        source_file = os.path.join(train_fake_path, filename)
        dest_file = os.path.join(new_train_fake_path, filename)
        shutil.copy(source_file, dest_file)

# Print new file count information
new_real_count = len(os.listdir(new_train_real_path))
new_fake_count = len(os.listdir(new_train_fake_path))

print("\nFile count information after processing:")
print(f"New training data folder - Real data: {new_real_count} files")
print(f"New training data folder - Fake data: {new_fake_count} files")

print("\nProcessing completed. New training data saved to:", "datasets/AVLips_new")

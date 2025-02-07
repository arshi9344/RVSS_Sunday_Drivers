import os
import csv

# Set the root directory containing images and subfolders
root_dir = os.path.join(".", "data", "Datasets_to_clean")
output_file = os.path.join(root_dir, "image_dataset.csv")

with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write header row: Subfolder, Filename, Label1, Label2, Label3, Label4.
    writer.writerow(["Subfolder", "Filename", "Label1", "Label2", "Label3", "Label4"])
    
    # Walk through root_dir and its subfolders.
    for current_path, directories, files in os.walk(root_dir):
        # Compute the relative subfolder path ('.' means root_dir itself)
        subfolder = os.path.relpath(current_path, root_dir)
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                # Assume the filename format is: label1_label2_label3_label4.ext
                parts = file.split("_")
                if len(parts) >= 4:
                    label1 = parts[0]
                    label2 = parts[1]
                    label3 = parts[2]
                    # Remove the extension from the fourth label
                    label4 = parts[3].split(".")[0]
                else:
                    label1 = label2 = label3 = label4 = ""
                writer.writerow([subfolder, file, label1, label2, label3, label4])

print("CSV file created successfully at:", output_file)

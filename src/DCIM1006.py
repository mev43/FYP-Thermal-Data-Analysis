import os
import sys
import csv
# Ensure project root is in sys.path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src import uniTThermalImage
import matplotlib.pyplot as plt

# Directory containing BMP images
image_dir = r"C:\Users\OEM\OneDrive\Documents\FYP Images\DCIM_1006"
output_csv = r"C:\Users\OEM\OneDrive\Documents\FYP Images\DCIM_1006\thermal_data.csv"

# Collect all BMP files
bmp_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.bmp')]

results = []
# Specify the point
point = (210, 130)

annotated_count = 0
valid_images = []
for bmp_file in bmp_files:
    img_path = os.path.join(image_dir, bmp_file)
    obj_uti = uniTThermalImage.UniTThermalImage()
    try:
        obj_uti.init_from_image(img_path)
        # Extract relevant data
        temp_at_point = None
        try:
            temp_at_point = obj_uti.raw_temp_np[point]
        except Exception as e:
            temp_at_point = 'N/A'
        results.append({
            'image': bmp_file,
            'min_temp': obj_uti.temp_min,
            'max_temp': obj_uti.temp_max,
            'center_temp': obj_uti.temp_center,
            'point_temp': temp_at_point,
            'units': obj_uti.temp_units
        })

        valid_images.append((bmp_file, obj_uti.raw_img_rgb_np))

        # Plot the first 5 successfully processed images with the point annotated
        if annotated_count < 5:
            plt.imshow(obj_uti.raw_img_rgb_np)
            plt.scatter(point[1], point[0], c='red', s=60, label=f'Point {point}')
            plt.title(f'{bmp_file} with point {point}')
            plt.legend()
            plt.savefig(os.path.join(image_dir, f'example_annotated_{annotated_count+1}.png'))
            plt.close()
            annotated_count += 1
    except Exception as e:
        print(f"Skipping {bmp_file}: {e}")

# Annotate the last 5 valid images
for i, (bmp_file, img_rgb_np) in enumerate(valid_images[-5:], 1):
    plt.imshow(img_rgb_np)
    plt.scatter(point[1], point[0], c='blue', s=60, label=f'Point {point}')
    plt.title(f'{bmp_file} with point {point} (last {i})')
    plt.legend()
    plt.savefig(os.path.join(image_dir, f'example_annotated_last_{i}.png'))
    plt.close()

# Write results to CSV

with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['image', 'min_temp', 'max_temp', 'center_temp', 'point_temp', 'units']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"Processed {len(results)} images. Data saved to {output_csv}")
print(f"Example images with point annotation saved as example_annotated_1.png to example_annotated_{annotated_count}.png and example_annotated_last_1.png to example_annotated_last_5.png in {image_dir}")

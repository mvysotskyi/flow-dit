import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_image_dimensions(directory):
    heights = []
    widths = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = cv2.imread(file_path)
                if img is not None:
                    h, w, _ = img.shape
                    heights.append(h)
                    widths.append(w)
            except Exception as e:
                print(f"Skipping {file_path}: {e}")

    return np.array(heights), np.array(widths)

def plot_distribution(data, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(save_path)
    plt.close()

def main(directory, output_dir="dataset_plots"):
    os.makedirs(output_dir, exist_ok=True)

    print("Scanning images...")
    heights, widths = get_image_dimensions(directory)
    
    if len(heights) == 0 or len(widths) == 0:
        print("No valid images found.")
        return

    aspect_ratios = widths / heights
    
    print("Generating plots...")
    plot_distribution(heights, "Height Distribution", "Height (px)", "Frequency", os.path.join(output_dir, "height_distribution.png"))
    plot_distribution(widths, "Width Distribution", "Width (px)", "Frequency", os.path.join(output_dir, "width_distribution.png"))
    plot_distribution(aspect_ratios, "Aspect Ratio Distribution", "Aspect Ratio (W/H)", "Frequency", os.path.join(output_dir, "aspect_ratio_distribution.png"))

    print(f"Plots saved in {output_dir}")

if __name__ == "__main__":
    image_directory = "/workspace/data/birds/"
    main(image_directory)


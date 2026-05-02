import random
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure Matplotlib does not display graphs in the GUI
plt.ioff()

# Define the output directory
OUTPUT_FOLDER = './static/outputs/'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # Ensure the folder exists


def overlay_segmentation(original_image, segmentation_mask, filename):
    """
    Overlays the segmentation mask on the original image.
    """
    save_path = os.path.join(OUTPUT_FOLDER, f"{filename}_overlay.png")

    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    plt.imshow(segmentation_mask, cmap='jet', alpha=0.5)  # Overlay with transparency
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_healing_time(prediction, filename):
    """
    Creates a bar chart representing wound healing time.
    """
    save_path = os.path.join(OUTPUT_FOLDER, f"{filename}_healing_chart.png")

    labels = ["Days", "Weeks", "Months", "Years"]
    values = [prediction, prediction / 7, prediction / 30, prediction / 365]

    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=values, palette="coolwarm")
    plt.ylabel("Time Units")
    plt.title("Predicted Wound Healing Time")
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def shap_explanation(metadata, filename):
    """
    Generates a SHAP explanation visualization.
    """
    save_path = os.path.join(OUTPUT_FOLDER, f"{filename}_shap.png")

    plt.figure(figsize=(8, 5))
    sns.barplot(x=[f"Feature {i+1}" for i in range(len(metadata[0]))], y=metadata[0], palette="viridis")
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Importance", fontsize=12)
    plt.title("SHAP Feature Importance", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_color_distribution(image, filename):
    """
    Plots histogram of RGB color distribution in the wound image.
    """
    save_path = os.path.join(OUTPUT_FOLDER, f"{filename}_histogram.png")

    color = ('b', 'g', 'r')  # Blue, Green, Red channels
    plt.figure(figsize=(6, 4))

    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col, linewidth=2)

    plt.title("Wound Color Distribution", fontsize=14)
    plt.xlabel("Pixel Intensity", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(["Blue", "Green", "Red"], fontsize=10)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {save_path}")
    return save_path



# --------------------------------
# **TEST MODE: Run Visualizations Standalone**
# --------------------------------
if __name__ == '__main__':
    print("Running visualization tests...")

    # Load a sample image (Replace with a real wound image path)
    sample_image_path = "/dataset/images/fusc_0002.jpg"  # Place a test image here
    if not os.path.exists(sample_image_path):
        print("No test image found. Creating a random test image.")
        sample_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)  # Random image
    else:
        sample_image = cv2.imread(sample_image_path)

    # Generate a random segmentation mask
    segmentation_mask = np.random.rand(sample_image.shape[0], sample_image.shape[1])  # Random segmentation

    # Generate random metadata
    metadata = np.array([[random.uniform(0, 1) for _ in range(10)]])  # 10 random features

    # Generate a random healing time prediction (in days)
    prediction = random.uniform(30, 365)  # Between 1 month and 1 year

    # Define a test filename
    test_filename = "test_case"

    # Call all visualizations
    overlay_segmentation(sample_image, segmentation_mask, test_filename)
    plot_healing_time(prediction, test_filename)
    shap_explanation(metadata, test_filename)
    plot_color_distribution(sample_image, test_filename)

    print("All test visualizations have been generated in 'static/outputs/'")

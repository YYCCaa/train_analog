import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

# --- Configuration ---
IMAGE_PATH = "sample.png"  # Change to your image path
BASE_PIXEL = (166,67,70)  # Fixed base pixel (R, G, B)

def plot_rgb_differences(image_path, base_pixel):
    img = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
    h, w, _ = img.shape

    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)

    channels = ["Red", "Green", "Blue"]
    colors = ["Reds", "Greens", "Blues"]

    fig = plt.figure(figsize=(18, 5))

    for i in range(3):
        diff = img[:, :, i] - base_pixel[i]

        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.plot_surface(X, Y, diff, cmap=colors[i], edgecolor="none", alpha=0.9)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel(f"Δ{channels[i]}")
        ax.set_title(f"{channels[i]} difference (base={base_pixel[i]})")

    plt.tight_layout()
    plt.savefig("rgb_diff_plot.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    plot_rgb_differences(path, BASE_PIXEL)

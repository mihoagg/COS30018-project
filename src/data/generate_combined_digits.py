import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import save_image
import random
import tensorflow as tf
import torch
import random
import os
from torchvision.utils import save_image

# Load MNIST
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

# Convert to torch
images = torch.tensor(x_train, dtype=torch.float32) / 255.0  # (60000, 28, 28)
labels = torch.tensor(y_train)

# Function to create combined image
def create_combined_image(images, labels, num_digits=5):
    selected_imgs = []
    selected_labels = []

    for _ in range(num_digits):
        idx = random.randint(0, len(images) - 1)
        img = images[idx]           # (28, 28)
        label = labels[idx].item()

        selected_imgs.append(img)
        selected_labels.append(str(label))

    # Concatenate horizontally (width = dim=1)
    combined = torch.cat(selected_imgs, dim=1)  # (28, 28*num_digits)

    # Add channel dimension for save_image
    combined = combined.unsqueeze(0)  # (1, H, W)

    return combined, "".join(selected_labels)

# Generate example
combined_img, label_str = create_combined_image(images, labels, num_digits=6)

os.makedirs("src/data/combined", exist_ok=True)
save_image(combined_img, f"src/data/combined/mnist_{label_str}.png")

print("Label:", label_str)

# Display image
plt.imshow(combined_img.squeeze(), cmap="gray")
plt.title(f"Label: {label_str}")
plt.axis("off")
plt.show()
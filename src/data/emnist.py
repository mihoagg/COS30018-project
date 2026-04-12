import numpy as np
from torchvision import datasets, transforms
from pathlib import Path

def load_emnist():
    """
    Loads the EMNIST 'balanced' dataset (47 classes) using torchvision.
    This is very robust and handles download/formatting automatically.
    """
    cache_dir = Path.home() / ".cache" / "torch_emnist"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading EMNIST (balanced) via torchvision...")
    train_set = datasets.EMNIST(
        root=str(cache_dir), 
        split='balanced', 
        train=True, 
        download=True
    )
    test_set = datasets.EMNIST(
        root=str(cache_dir), 
        split='balanced', 
        train=False, 
        download=True
    )
    
    # Extract data as numpy arrays
    x_train = train_set.data.numpy()
    y_train = train_set.targets.numpy()
    x_test = test_set.data.numpy()
    y_test = test_set.targets.numpy()
    
    # EMNIST is stored transposed (W, H). Transpose back to (H, W) to match MNIST.
    # We use swapaxes to transform (N, 28, 28) properly.
    x_train = np.swapaxes(x_train, 1, 2)
    x_test = np.swapaxes(x_test, 1, 2)

    print(f"EMNIST Loaded: {len(x_train)} training samples.")
    return (x_train, y_train), (x_test, y_test)

def get_label_mapping(source="mnist"):
    """Returns a list mapping indices to characters for the selected dataset."""
    if source == "mnist":
        return [str(i) for i in range(10)]
    
    if source == "emnist":
        # EMNIST Balanced mapping (47 classes)
        # 0-9: Digits
        # 10-35: Uppercase A-Z
        # 36-46: Lowercase a, b, d, e, f, g, h, n, q, r, t
        mapping = [str(i) for i in range(10)]
        mapping += [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        
        # Specific lowercase letters in 'balanced' split
        lowercase_indices = [ord('a'), ord('b'), ord('d'), ord('e'), ord('f'), 
                             ord('g'), ord('h'), ord('n'), ord('q'), ord('r'), ord('t')]
        mapping += [chr(i) for i in range(ord('a'), ord('z') + 1) if i in lowercase_indices]
        
        return mapping
        
    raise ValueError(f"Unknown source: {source}")

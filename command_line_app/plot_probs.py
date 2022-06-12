from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def plot_probs(image_path, top_ps, top_k_cats, top_k=1):
    image = Image.open(image_path)
    fig, (ax1, ax2) = plt.subplots(figsize=(10,10), ncols=2);
    ax1.imshow(image)
    ax1.axis('off')
    ax2.barh(np.arange(top_k), top_ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(top_k))
    ax2.set_yticklabels(top_k_cats, size='small')
    ax2.set_title('Category Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()

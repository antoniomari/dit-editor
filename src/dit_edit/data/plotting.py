import matplotlib.pyplot as plt

from dit_edit.data.benchmark_data import BenchmarkExample


def plot_results(example: BenchmarkExample):
    """Plot background, foreground and all results in a single row."""

    # Determine how many images we have (base images + results)
    num_images = 3  # bg, fg, result, final_mask
    if hasattr(example, "tf_icon_image") and example.tf_icon_image:
        num_images += 1
    if hasattr(example, "kvedit_image") and example.kvedit_image:
        num_images += 1

    # Create a new figure with a single row
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 5, 5))

    # Display base images
    axes[0].imshow(example.bg_image)
    axes[0].set_title("Background")
    axes[0].axis("off")

    axes[1].imshow(example.fg_image)
    axes[1].set_title("Foreground")
    axes[1].axis("off")

    axes[2].imshow(example.result_image)
    axes[2].set_title("Naive Composition")
    axes[2].axis("off")

    # Display additional results if available
    idx = 3
    if hasattr(example, "tf_icon_image") and example.tf_icon_image:
        axes[idx].imshow(example.tf_icon_image)
        axes[idx].set_title("TF-Icon Result")
        axes[idx].axis("off")
        idx += 1

    if hasattr(example, "kvedit_image") and example.kvedit_image:
        axes[idx].imshow(example.kvedit_image)
        axes[idx].set_title("KVEdit Result")
        axes[idx].axis("off")

    # Set the title
    fig.suptitle(
        f"Category: {example.category}, Image: {example.image_number}\nPrompt: {example.prompt}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


def plot_sample(example: BenchmarkExample):
    """
    Plot a sample from the benchmark example with masks in a separate row.
    """
    # Create a new figure with 2 rows
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))

    # First row: Original images
    ax[0, 0].imshow(example.bg_image)
    ax[0, 0].set_title("Background Image")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(example.fg_image)
    ax[0, 1].set_title("Foreground Image")
    ax[0, 1].axis("off")

    ax[0, 2].imshow(example.result_image)
    ax[0, 2].set_title("Result Image")
    ax[0, 2].axis("off")

    # Second row: Mask images
    ax[1, 0].imshow(example.target_mask, cmap="gray")
    ax[1, 0].set_title("Foreground Mask")
    ax[1, 0].axis("off")

    ax[1, 1].imshow(example.fg_mask, cmap="gray")
    ax[1, 1].set_title("Target Mask")
    ax[1, 1].axis("off")

    ax[1, 2].imshow(example.final_mask, cmap="gray")
    ax[1, 2].set_title("Final Mask")
    ax[1, 2].axis("off")

    # Set the title
    fig.suptitle(
        f"Category: {example.category}, Image: {example.image_number}\nPrompt: {example.prompt}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()

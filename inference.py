import matplotlib.pyplot as plt
from torchvision import io, utils
import lightly_train

# 1. Load model and predict
model = lightly_train.load_model("experiments/rect_convnext-small(moose_elk_6.19)/exported_models/exported_best.pt")
results = model.predict("images/1303.jpg", threshold=0.7)

# 2. Format labels to include the confidence score
labels_with_scores = [
    f"{model.classes[label.item()]}: {score.item():.2f}" 
    for label, score in zip(results["labels"], results["scores"])
]

# 3. Visualize predictions with custom sizes
image_with_boxes = utils.draw_bounding_boxes(
    image=io.read_image("images/1303.jpg"),
    boxes=results["bboxes"],
    labels=labels_with_scores,
    width=10,          # Controls the thickness of the bounding box lines
    font_size=50,
    colors="red"  # Controls the color of the bounding box lines
    # font="arial.ttf" # Optional: specify a path to a TrueType font file if the default font is too small
)

fig, ax = plt.subplots(figsize=(30, 30))
ax.imshow(image_with_boxes.permute(1, 2, 0))
fig.savefig("images/predicted moose?.png")
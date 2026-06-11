import supervision as sv
import lightly_train
from tqdm import tqdm
from supervision.metrics import MeanAveragePrecision

checkpoint = "experiments/rect_convnext-small(new_5.28)/exported_models/exported_best.pt"
data_yaml_path = "/home/higo522/RMNP/new_dataset(5.28)/test_data.yaml"
images_directory_path = "/home/higo522/RMNP/new_dataset(5.28)/val/images"
annotations_directory_path = "/home/higo522/RMNP/new_dataset(5.28)/val/labels"

model = lightly_train.load_model(checkpoint)
ds = sv.DetectionDataset.from_yolo(images_directory_path, annotations_directory_path, data_yaml_path)

targets, predictions = [], []
for path, image, annotations in tqdm(ds):
    raw_output = model.predict(path, threshold=0)
    detections = sv.Detections(
        xyxy=raw_output["bboxes"].cpu().numpy(),
        confidence=raw_output["scores"].cpu().numpy(),
        class_id=raw_output["labels"].cpu().numpy().astype(int)
    )
    targets.append(annotations)
    predictions.append(detections)

cm = sv.ConfusionMatrix.from_detections(
    predictions=predictions, targets=targets,
    classes=['Moose'], conf_threshold=0.65, iou_threshold=0.5,
).matrix
print(cm)

TP_M = cm[0, 0]
FP_M = cm[:, 0].sum() - TP_M
FN_M = cm[0, :].sum() - TP_M

precision_m = TP_M / (TP_M + FP_M + 1e-9)
recall_m    = TP_M / (TP_M + FN_M + 1e-9)
F1_m        = 2 * precision_m * recall_m / (precision_m + recall_m + 1e-9)

map_result = MeanAveragePrecision().update(predictions, targets).compute()

print(f"MOOSE: Precision={precision_m:.4f}, Recall={recall_m:.4f}, F1={F1_m:.4f}, mAP50={map_result.ap_per_class[0,0]:.3f}, mAP50-95={map_result.ap_per_class[0].mean():.3f}")
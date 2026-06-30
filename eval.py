import supervision as sv
import lightly_train
from tqdm import tqdm
from supervision.metrics import MeanAveragePrecision

checkpoint = "experiments/rect_convnext-small(moose_elk_6.19)/exported_models/exported_best.pt"
data_yaml_path = "/home/higo522/RMNP/moose_elk_6.19/test_data.yaml"
images_directory_path = "/home/higo522/RMNP/moose_elk_6.19/val/images"
annotations_directory_path = "/home/higo522/RMNP/moose_elk_6.19/val/labels"

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
    classes=['Moose', 'Elk'], conf_threshold=0.65, iou_threshold=0.5,
).matrix
print(cm)

TP_M = cm[0, 0]
FP_M = cm[:, 0].sum() - TP_M
FN_M = cm[0, :].sum() - TP_M

precision_m = TP_M / (TP_M + FP_M + 1e-9)
recall_m    = TP_M / (TP_M + FN_M + 1e-9)
F1_m        = 2 * precision_m * recall_m / (precision_m + recall_m + 1e-9)

TP_E = cm[1, 1]
FP_E = cm[:, 1].sum() - TP_E
FN_E = cm[1, :].sum() - TP_E

precision_e = TP_E / (TP_E + FP_E + 1e-9)
recall_e    = TP_E / (TP_E + FN_E + 1e-9)
F1_e        = 2 * precision_e * recall_e / (precision_e + recall_e + 1e-9)

map_result = MeanAveragePrecision().update(predictions, targets).compute()

print(f"MOOSE: Precision={precision_m:.4f}, Recall={recall_m:.4f}, F1={F1_m:.4f}, mAP50={map_result.ap_per_class[0,0]:.3f}, mAP50-95={map_result.ap_per_class[0].mean():.3f}")
print(f"ELK:   Precision={precision_e:.4f}, Recall={recall_e:.4f}, F1={F1_e:.4f}, mAP50={map_result.ap_per_class[1,0]:.3f}, mAP50-95={map_result.ap_per_class[1].mean():.3f}")
import lightly_train

NAME = "rect_convnext-small(moose_elk_6.19)"

def main():
    lightly_train.train_object_detection(
        out=f"experiments/{NAME}",
        model="dinov3/convnext-small-ltdetr-coco",
        batch_size=4,
        steps=20000,
        accelerator="gpu",
        devices=1,
        overwrite=True,
        data={
            "format": "yolo",
            "path": "/home/higo522/RMNP/moose_elk_6.19",
            "train": "train/images",
            "val": "val/images",
            "names": {0: "moose", 1: "elk"},
        },
        logger_args={
            "wandb": {
                "project": "rmnp-detection",
                "name": NAME,
                "log_model": False,
            },
            "val_every_num_steps": 1000,
        },
        save_checkpoint_args={
            "save_last": False,
            "save_best": True,
        },
        transform_args={
            "image_size": (480, 1280),
            "scale_jitter": None, # only works for square images, so we disable it for our rectangular input
        },
    )

if __name__ == "__main__":
    main()
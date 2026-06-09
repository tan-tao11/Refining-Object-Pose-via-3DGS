import argparse
import cv2
import os
import warnings
from tqdm import tqdm
import os.path as osp
from ultralytics import YOLO
from .FastSAM.tools import *
from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
from torchvision.ops import box_convert
import ast

warnings.filterwarnings("ignore")

def main(
        imgs_path,
        text,
        save_path,
        imgsz=[512, 512],
        sam_model_path='tools/mask/checkpoints/FastSAM-x.pt',
        device='cuda',
        retina=True,
        iou=0.9,
        conf=0.4,
):
    # Build Fast-SAM Model
    # ckpt_path = "/comp_robot/rentianhe/code/Grounded-Segment-Anything/FastSAM/FastSAM-x.pt"
    sam_model = YOLO(sam_model_path)

    # Build GroundingDINO Model
    groundingdino_config = "tools/mask/GroundingDINO_SwinT_OGC.py"
    groundingdino_ckpt_path = "tools/mask/checkpoints/groundingdino_swint_ogc.pth"
    dino_model = load_model(groundingdino_config, groundingdino_ckpt_path)

    files = os.listdir(imgs_path)
    images = [file for file in files if file.endswith('.png')]
    for image_file in tqdm(images):
        img_path = osp.join(imgs_path, image_file)
        # Image Path
        img_path = img_path
        text = text

        # path to save img
        basename = os.path.basename(img_path).split(".")[0]

        results = sam_model(
            img_path,
            imgsz=imgsz,
            device=device,
            retina_masks=retina,
            iou=iou,
            conf=conf,
            max_det=100,
        )

        image_source, image = load_image(img_path)
        

        boxes, logits, phrases = predict(
            model=dino_model,
            image=image,
            caption=text,
            box_threshold=0.3,
            text_threshold=0.25,
            device=device,
        )

        # Grounded-Fast-SAM
        ori_img = cv2.imread(img_path)
        ori_h = ori_img.shape[0]
        ori_w = ori_img.shape[1]

        # Save each frame due to the post process from FastSAM
        boxes = boxes * torch.Tensor([ori_w, ori_h, ori_w, ori_h])
        print(f"Detected Boxes: {len(boxes)}")
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy().tolist()
        full_mask = np.zeros([ori_h, ori_w])
        for box_idx in range(len(boxes)):
            mask, _ = box_prompt(
                results[0].masks.data,
                boxes[box_idx],
                ori_h,
                ori_w,
            )
            # annotations = np.array([mask])
            # # img_array = fast_process(
            # #     annotations=annotations,
            # #     img_path=img_path,
            # #     output=save_path,
            # #     mask_random_color=True,
            #     bbox=boxes[box_idx],
            # )
            full_mask[mask==1] = 1
            # mask = mask * 255
        full_mask = full_mask * 255
        cv2.imwrite(osp.join(save_path, image_file), full_mask.astype(np.uint8))

def predict_masks_fastsam(image_folder,  text_prompt, output_dir_mask):
    # files = os.listdir(image_folder)
    # images = [file for file in files if file.endswith('.png')]

    main(image_folder, text_prompt, output_dir_mask)


import math
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import Callable
import cv2
import torch
from torch import nn
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from exordium.video.bb import iou_xyxy


def inference_img(model: nn.Module, image_np: np.ndarray, device: str = "cuda:0") -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    img = Image.fromarray(image_np)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)

    with torch.no_grad():
        prediction = model(batch_t.to(device))

    scores = prediction[0]['scores'].cpu().detach().numpy()
    bbs_xyxy = prediction[0]['boxes'].cpu().detach().numpy()
    labels = prediction[0]['labels'].cpu().detach().numpy()
    return bbs_xyxy, scores, labels


def generate_mask(image_size: tuple[int, int], grid_size: tuple[int, int], prob_thr: float) -> np.ndarray:
    image_w, image_h = image_size
    grid_w, grid_h = grid_size
    cell_w, cell_h = math.ceil(image_w / grid_w), math.ceil(image_h / grid_h)
    up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h
    mask = (np.random.uniform(0, 1, size=(grid_h, grid_w)) < prob_thr).astype(np.float32)
    mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
    offset_w = np.random.randint(0, cell_w)
    offset_h = np.random.randint(0, cell_h)
    mask = mask[offset_h:offset_h + image_h, offset_w:offset_w + image_w]
    return mask


def mask_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return ((image.astype(np.float32) / 255 * np.dstack([mask] * 3)) * 255).astype(np.uint8)


def plot_samples(image: np.ndarray,
                 inference_img: Callable[[nn.Module, np.ndarray],tuple[np.ndarray, np.ndarray, np.ndarray]],
                 model: nn.Module,
                 output_path: str,
                 prob_thr: float = 0.5) -> None:
    np.random.seed(0)
    image_h, image_w = image.shape[:2]
    res = image.copy()

    bbs_xyxy, scores, labels = inference_img(model, image)
    for ind, (bb_xyxy, score, label) in enumerate(zip(bbs_xyxy, scores, labels)):
        if score < 0.5: continue
        box = np.round(bb_xyxy).astype(int)
        c = (0, 255, 0) if ind == 0 else (0, 0, 0)
        cv2.rectangle(res, box[:2], box[2:], c, 5)
        print('predicted label:', label, 'bb_xyxy', box, 'score:', score)
    cv2.imwrite(str(Path(output_path).parent / f'{Path(output_path).stem}_orig.png'), res)

    images = []
    for _ in range(25):
        mask = generate_mask(image_size=(image_w, image_h),
                             grid_size=(16, 16),
                             prob_thr=prob_thr)
        masked = mask_image(image, mask)
        bb_xyxy, scores, _ = inference_img(model, masked)
        res = masked.copy()

        for bb, score in zip(bb_xyxy, scores):
            if score < prob_thr: continue
            box = np.round(bb).astype(int)
            cv2.rectangle(res, box[:2], box[2:], (0, 255, 0), 5)

        images.append(res)

    fig = plt.figure(figsize=(15, 10))
    axes = fig.subplots(5, 5)
    for i in range(5):
        for j in range(5):
            axes[i][j].imshow(images[i * 5 + j][:, :, ::-1])
            axes[i][j].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)


def save_saliency_map(image: np.ndarray, bb_xyxy: np.ndarray, saliency_map: np.ndarray, output_path: str) -> None:
    cv2.rectangle(image, tuple(bb_xyxy[:2]), tuple(bb_xyxy[2:]), (0, 255, 0), 5)
    plt.figure(figsize=(7, 7))
    plt.imshow(image[:, :, ::-1])
    plt.imshow(saliency_map, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.savefig(output_path)


def drise(image: np.ndarray,
          inference_image: Callable[[nn.Module, np.ndarray], np.ndarray],
          model: nn.Module,
          target_class_index: int,
          target_bb_xyxy: np.ndarray,
          prob_thr: float = 0.25,
          grid_size: tuple[int, int] = (16, 16),
          n_masks: int = 5000,
          seed: int = 0,
          output_path: str | None = None,
          verbose: float = True) -> np.ndarray:

    np.random.seed(seed)
    image_h, image_w = image.shape[:2]

    saliency_map = np.zeros((image_h, image_w), dtype=np.float32)
    for _ in tqdm(range(n_masks), total=n_masks, desc="D-RISE", disable=not verbose):
        mask = generate_mask(image_size=(image_w, image_h),
                             grid_size=grid_size,
                             prob_thr=prob_thr)
        masked = mask_image(image, mask)
        boxes, scores, labels = inference_image(model, masked)
        score = max([iou_xyxy(target_bb_xyxy, box) * score for box, score, label in zip(boxes, scores, labels) if label == target_class_index], default=0)
        saliency_map += mask * score

    if output_path is not None:
        save_saliency_map(image, target_bb_xyxy, saliency_map, output_path)

    return saliency_map


if __name__ == '__main__':
    from torchvision import models
    image_path = 'data/images/cat_tie.jpg'
    img = Image.open(image_path).convert('RGB')
    image_np = cv2.imread(image_path)
    frcnn = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    frcnn.eval()
    frcnn.to(torch.device("cuda:0"))
    saliency_map = drise(image=image_np, inference_image=inference_img, model=frcnn, target_class_index=32, target_bb_xyxy=np.array([500, 1005, 937, 1231]), output_path='data/tmp/saliency.png', n_masks=1000, prob_thr=0.25)
    plot_samples(image_np, inference_img, frcnn, 'data/tmp/cat_tie_vis.png')
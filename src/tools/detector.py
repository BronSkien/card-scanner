import numpy as np
import torch
import sys
import functools
import os
import cv2
import mmcv
import warnings
from mmdet.apis import init_detector, inference_detector

# Suppress warnings
warnings.filterwarnings("ignore")


def disable_print(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Redirect stdout to suppress print output
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            result = func(*args, **kwargs)
        # Restore stdout
        sys.stdout = sys.__stdout__
        return result
    return wrapper


@disable_print
class Detector:
    def __init__(self, model_name, weights_url):
        # For MMDetection 2.20.0, we need to use a config file
        # We'll use a standard config for instance segmentation
        config_file = self._get_config_for_model(model_name)
        
        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize the detector
        self.model = init_detector(config_file, weights_url, device=self.device)
        print(f"Model initialized: {model_name}")
    
    def _get_config_for_model(self, model_name):
        # Map model names to their config files
        # For rtmdet-ins_tiny, we'll use the RTMDet-Ins tiny config
        if 'rtmdet-ins_tiny' in model_name:
            from mmdet.apis import get_config
            config = get_config('rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py')
            return config
        else:
            # Default to a standard instance segmentation config
            return 'mask_rcnn_r50_fpn_1x_coco.py'

    @disable_print
    def detect_objects(self, img_path, scoreThreshold=0.5):
        # Load the image
        img = mmcv.imread(img_path)
        
        # Perform inference
        result = inference_detector(self.model, img)
        
        # Process results
        if hasattr(result, 'pred_instances'):
            # MMDetection 3.x style results
            bboxes = result.pred_instances.bboxes.cpu().numpy()
            scores = result.pred_instances.scores.cpu().numpy()
            masks = result.pred_instances.masks.cpu().numpy() if hasattr(result.pred_instances, 'masks') else None
        else:
            # MMDetection 2.x style results (list of arrays)
            # Assuming result[0] contains bbox+score and result[1] contains masks
            if isinstance(result, tuple):
                bbox_result, mask_result = result
                # Flatten the bbox results for all classes
                bboxes = np.vstack([bbox_result[i] for i in range(len(bbox_result))])
                scores = bboxes[:, 4] if bboxes.shape[1] > 4 else np.ones(bboxes.shape[0])
                bboxes = bboxes[:, :4]
                
                # Process masks if available
                if mask_result is not None:
                    masks = np.stack([mask_result[i] for i in range(len(mask_result))], axis=0)
                else:
                    masks = None
            else:
                # Just bbox results
                bboxes = np.vstack([result[i] for i in range(len(result))])
                scores = bboxes[:, 4] if bboxes.shape[1] > 4 else np.ones(bboxes.shape[0])
                bboxes = bboxes[:, :4]
                masks = None
        
        # If no masks, create binary masks from bounding boxes
        if masks is None:
            img_h, img_w = img.shape[:2]
            masks = []
            for bbox in bboxes:
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                x1, y1, x2, y2 = map(int, bbox)
                mask[y1:y2, x1:x2] = 1
                masks.append(mask)
            masks = np.array(masks)
        
        # Create detection objects
        detections = []
        for i in range(len(scores)):
            if scores[i] > scoreThreshold:
                detection = {
                    'bbox': list(map(int, bboxes[i])),  # convert to int
                    'score': float(scores[i]),
                    'mask': masks[i] if masks is not None else None
                }
                detections.append(detection)
        
        return detections

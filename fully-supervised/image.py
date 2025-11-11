import os
import random
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import cv2

def generate_segmentation_image(config):
    """
    Generates an image comparing predicted segmentation masks with the original images
    and ground truth. Displays and saves a 4-column plot with:
        1. Original Image
        2. Predicted Binary Mask
        3. Overlay of Prediction on Original
        4. Ground Truth Trimap (if found)
    Args:
        config (module or object): Contains configuration attributes like model path,
                                   image size, device, and output paths.
    """

    def dark_lighten_mask_region(original_img, mask_01, alpha_light=0.5, alpha_dark=0.5):
        if not isinstance(original_img, np.ndarray):
            original_img = np.array(original_img)  
        overlay = original_img.astype(np.float32)

        black = np.array([255,255,255], dtype=np.float32)
        white = np.array([  0,  0,  0], dtype=np.float32)

        region_fg = (mask_01 == 1)
        overlay[region_fg] = (1 - alpha_light)*overlay[region_fg] + alpha_light*white

        region_bg = (mask_01 == 0)  
        overlay[region_bg] = (1 - alpha_dark)*overlay[region_bg] + alpha_dark*black

        return overlay.astype(np.uint8)
    
    if os.path.exists(config.PRED_PLOT_PATH):
        print("Prediction image already exists, skipping image generation.")
        return

    model = torch.load(config.MODEL_PATH, weights_only=False).to(config.DEVICE)
    model.eval()

    with open(config.TEST_IMGS_PATH, 'rb') as f:
        test_imgs = pickle.load(f)

    indices = random.sample(range(len(test_imgs)), 3)
    selected_imgs = [test_imgs[idx] for idx in indices]

    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor()
    ])

    num_samples = len(selected_imgs)
    figure, ax = plt.subplots(num_samples, 4,
                              figsize=(16, 4*num_samples),
                              subplot_kw={'xticks': [], 'yticks': []})
    if num_samples == 1:
        ax = np.expand_dims(ax, axis=0)

    with torch.no_grad():
        for i, img_path in enumerate(selected_imgs):

            #first col
            orig_img = Image.open(img_path).convert('RGB')
            orig_img_disp = transforms.Resize(config.IMAGE_SIZE)(orig_img)

            img_tensor = transform(orig_img).to(config.DEVICE)
            img_input  = img_tensor.unsqueeze(0)

            #second col
            if hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Sequential):
                output = model(img_input)['out']
            else:
                output = model(img_input)

            pred_logits = output.squeeze(0)  
            pred_prob   = torch.sigmoid(pred_logits).cpu().numpy()
            pred_prob   = cv2.GaussianBlur(pred_prob, (5,5), 0)

            pred_mask   = (pred_prob >= 0.5).astype(np.uint8)*255
            if pred_mask.ndim == 3:
                pred_mask = pred_mask.squeeze(0)  

            
            ax[i,0].imshow(orig_img_disp)
            ax[i,0].set_title("Original Image")
            ax[i,0].axis("off")

            inv_mask = 255 - pred_mask  # so FG->0, BG->255
            ax[i,1].imshow(inv_mask, cmap="gray", vmin=0, vmax=255)
            ax[i,1].set_title("Predicted Mask")
            ax[i,1].axis("off")

            #third col
            mask_01 = (pred_mask // 255).astype(np.uint8)
            overlay_img = dark_lighten_mask_region(
                original_img=orig_img_disp,
                mask_01=mask_01,
                alpha_light=0.5,  
                alpha_dark=0.5   
            )
            ax[i,2].imshow(overlay_img)
            ax[i,2].set_title("Predict Mask Overlay Image")
            ax[i,2].axis("off")

            #fourth col
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            trimap_path = os.path.join("ground-truth", base_name + ".png")
            if os.path.exists(trimap_path):
                trimap_color_disp = Image.open(trimap_path).resize(config.IMAGE_SIZE)
                ax[i,3].imshow(trimap_color_disp)
                ax[i,3].set_title("Ground Truth")
            else:
                ax[i,3].text(0.5, 0.5, "No Trimap", fontsize=12, ha="center")
            ax[i,3].axis("off")

        #show image
        plt.tight_layout()
        figure.savefig(config.PRED_PLOT_PATH, bbox_inches='tight')
        plt.show()
        print("4-column figure saved to:", config.PRED_PLOT_PATH)

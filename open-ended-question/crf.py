import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

def apply_crf(image, softmax_output, num_iterations=5):
    """
    Apply DenseCRFto refine segmentation predictions.
    Args:
        image (np.ndarray): Input RGB image as a NumPy array of shape (H, W, 3).
        softmax_output (np.ndarray): Softmax probability map of shape (num_classes, H, W),
                                     where each channel corresponds to the class probability map.
        num_iterations (int): Number of CRF inference iterations to run (default: 5).
    Returns:
        np.ndarray: Refined segmentation mask of shape (H, W), with integer class labels.
    """

    H, W = image.shape[:2]
    num_classes = softmax_output.shape[0]
    
    # Create a DenseCRF2D object.
    d = dcrf.DenseCRF2D(W, H, num_classes)
    
    # Convert softmax output to unary energy.
    unary = utils.unary_from_softmax(softmax_output)
    d.setUnaryEnergy(unary)
    
    # Add pairwise Gaussian features (spatial smoothness).
    d.addPairwiseGaussian(sxy=3, compat=3)
    
    # Ensure the image is C-contiguous in memory.
    image_contiguous = np.ascontiguousarray(image)
    
    # Add pairwise bilateral features (appearance-based consistency).
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image_contiguous, compat=10)
    
    # Run CRF inference.
    Q = d.inference(num_iterations)
    
    # Get the final segmentation mask.
    refined_seg = np.argmax(Q, axis=0).reshape((H, W))
    
    return refined_seg

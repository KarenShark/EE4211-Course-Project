import torch
import torch.nn.functional as F

class GradCAM:
    """
    Method to compute Grad-CAM images for convolutional neural networks.
    Args:
        model (torch.nn.Module): The pretrained model to analyze.
        target_layer (torch.nn.Module): The convolutional layer to target for activation maps.
    """

    def __init__(self, model, target_layer):
        model.eval()
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_images):
        """
        Generate Grad-CAM heatmaps for a batch of input images.
        Args:
            input_images (torch.Tensor): A batch of input images of shape [B, C, H, W].
        Returns:
            list of np.ndarray: A list of normalized 2D CAM heatmaps, one for each image.
        """
            
        model_output = self.model(input_images)
        class_indices = model_output.argmax(dim=1)
        self.model.zero_grad()

        cams = []

        for i in range(input_images.size(0)):
            self.model.zero_grad() 
            #get class score here 
            class_score = model_output[i, class_indices[i]]

            #bcakward pass 
            class_score.backward(retain_graph=(i < input_images.size(0)-1))

            #gradients and activations
            gradients = self.gradients[i]
            activations = self.activations[i]

            #weights for each channel 
            weights = torch.mean(gradients, dim=[1, 2], keepdim=True)

            #gradcam for one image
            cam = torch.sum(weights * activations, dim=0, keepdim=True)

            #apply reLu to image 
            cam = F.relu(cam)

            #normalisation 
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)

            # Append the CAM for this image to the list
            cams.append(cam.squeeze().cpu().numpy())

        # Return a list of CAMs for each image in the batch
        return cams
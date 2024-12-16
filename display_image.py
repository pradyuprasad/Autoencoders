import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image


def plot_image(image, label=None):
   plt.figure(figsize=(6, 6))
   
   if isinstance(image, torch.Tensor):
       image = image.numpy()
       if len(image.shape) == 3 and image.shape[0] in [1, 3]:
           image = image.transpose(1, 2, 0)
       if image.shape[-1] == 1:
           image = image.squeeze()
           
   elif isinstance(image, Image.Image):
       image = np.array(image)
   
   if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[-1] == 1):
       plt.imshow(image, cmap='gray')
   else:
       plt.imshow(image)
   
   if label is not None:
       plt.title(f'Label: {label}')
   
   plt.axis('off')
   plt.show()
   plt.close()

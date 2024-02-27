# %%
import os
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import skimage
from skimage.feature import peak_local_max
from skimage.filters import gaussian

# %%
from PIL import Image
import cv2
import numpy as np
img = cv2.imread("C:/Users/WitteVerheul/Desktop/Slangenburg_DSM_v1.tif", -1)
print(img.shape)
img[img == -10000] = img[img != -10000].min()
min_val = img.min()
max_val = img.max()
img_norm = (img - min_val) / (max_val - min_val) * 2**16 - 1
img_norm = img_norm.astype(np.uint16)

print(min_val)
print(max_val)
# %%

cv2.imshow("", img_norm[10000:11000, 10000:12000])
cv2.waitKey(0)
#im = Image.open('Z:/Clients/Skogran/Demo_Finland2023/Orthomosaic/20230608-ClearTimber-Analytics-Online_planning_1686302203--dsm-zip.tif')
#im.show()

# %%
img_crop = img_norm[10000:11000, 10000:12000]
radius = 25
kernel = np.zeros((2*radius+1, 2*radius+1))
y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
mask2 = x**2 + y**2 <= radius**2
kernel[mask2] = 1

# image_max is the dilation of im with a 20*20 structuring element
# It is used within peak_local_max function
image_max = ndi.maximum_filter(img, footprint=kernel, mode='constant')

# Gaussian Filter for better results
image_gaus = gaussian(image_max)

# Comparison between image_max and im to find the coordinates of local maxima
coordinates = peak_local_max(image_gaus, footprint=kernel)


# %%

# display results
fig, axes = plt.subplots(3, 1, figsize=(50, 20), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(img_crop, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('Original')

ax[1].imshow(image_max, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('Maximum filter')

ax[2].imshow(img_crop, cmap=plt.cm.gray)
ax[2].autoscale(False)
ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
ax[2].axis('off')
ax[2].set_title('Peak local max')

fig.tight_layout()

plt.show()
# %%
# Define helper fucntion to create cropped images
def crop_around_point(image, x, y, crop_size=512):
    half_size = crop_size // 2
    left = max(0, x - half_size)
    upper = max(0, y - half_size)
    right = min(image.width, x + half_size)
    lower = min(image.height, y + half_size)

    return image.crop((left, upper, right, lower))

# Output location for the images
output_dir = "C:\Users\WitteVerheul\Desktop\Peak_maximum_dataset"
os.makedirs(output_dir, exist_ok=True)

# Crop and save images
for i, (x, y) in enumerate(coordinates):
    cropped_image = crop_around_point(image_gaus, x, y)
    output_path = os.path.join(output_dir, f"cropped_image_{i + 1}.jpg")
    cropped_image.save(output_path)
    print(f"Image {i + 1} cropped and saved to {output_path}")

#%%
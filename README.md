# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!

## Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.

## Program:
````
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the Face Image
faceImage = cv2.imread('dharsh.jpg')
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")
````

<img width="281" height="37" alt="image" src="https://github.com/user-attachments/assets/0057ab88-c8fe-4d4f-88b8-40d8f8025ad0" />

```
#resized_faceImage.shape
faceImage.shape
```

<img width="547" height="592" alt="image" src="https://github.com/user-attachments/assets/8a64ba68-0276-4523-a4a5-0f988bef1b3e" />

```
glassPNG = cv2.imread('sunglass.jpg',-1)
plt.imshow(glassPNG[:,:,::-1]);plt.title("glassPNG")
# Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG,(190,50))
```

```
# Separate the Color and alpha channels
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]
# Display the images for clarity
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(glassBGR[:,:,::-1]);plt.title('Sunglass Color channels');
plt.subplot(122);plt.imshow(glassMask1,cmap='gray');plt.title('Sunglass Alpha channel');
```

<img width="845" height="155" alt="image" src="https://github.com/user-attachments/assets/9dc60a93-0ab6-47f7-9236-4e5336b145b1" />

```
import cv2
import numpy as np
import matplotlib.pyplot as plt

faceWithGlassesNaive = faceImage.copy()

# Target position
x, y = 118, 200

# Desired size
target_w, target_h = 160, 60

# Resize glasses
glassResized = cv2.resize(glassBGR, (target_w, target_h))

# Get image dimensions
h, w = faceWithGlassesNaive.shape[:2]

# Clip if overlay goes out of bounds
end_x = min(x + target_w, w)
end_y = min(y + target_h, h)

# Adjust overlay size to match clipped ROI
glassResized = glassResized[:end_y - y, :end_x - x]

# Overlay
faceWithGlassesNaive[y:end_y, x:end_x] = glassResized

plt.imshow(faceWithGlassesNaive[..., ::-1])
plt.axis("off")
plt.show()
```

<img width="222" height="389" alt="image" src="https://github.com/user-attachments/assets/274a8dc4-59a7-40a6-b7a8-55f5bfca5d98" />

```
# Assuming glassPNG has alpha channel
glassAlpha = glassPNG[..., 3] / 255.0
glassBGR = glassPNG[..., :3]

# Resize
glassBGR = cv2.resize(glassBGR, (target_w, target_h))
glassAlpha = cv2.resize(glassAlpha, (target_w, target_h))

# Extract eye region from face
eyeRoi = faceImage[y:y+target_h, x:x+target_w].copy()

# Masked eye and glasses
maskedEye = eyeRoi * (1 - glassAlpha[..., np.newaxis])
maskedGlass = glassBGR * glassAlpha[..., np.newaxis]
eyeRoiFinal = maskedEye + maskedGlass

# Convert to uint8 for display
maskedEye_disp = np.clip(maskedEye.astype(np.uint8), 0, 255)
maskedGlass_disp = np.clip(maskedGlass.astype(np.uint8), 0, 255)
eyeRoiFinal_disp = np.clip(eyeRoiFinal.astype(np.uint8), 0, 255)

# Display 3-panel intermediate results
plt.figure(figsize=[20,20])
plt.subplot(131)
plt.imshow(maskedEye_disp[..., ::-1])
plt.title("Masked Eye Region")
plt.axis("off")

plt.subplot(132)
plt.imshow(maskedGlass_disp[..., ::-1])
plt.title("Masked Sunglass Region")
plt.axis("off")

plt.subplot(133)
plt.imshow(eyeRoiFinal_disp[..., ::-1])
plt.title("Augmented Eye and Sunglass")
plt.axis("off")
plt.show()
```
<img width="826" height="134" alt="image" src="https://github.com/user-attachments/assets/ee4d0a7d-5c54-41b2-bd6b-4ec8ad4bd497" />

```
# Create a copy of the original face
faceWithGlasses = faceImage.copy()

# Apply alpha blending of sunglasses (example)
x, y = 102, 195
target_w, target_h = 192, 80

# Image dimensions
h, w = faceWithGlasses.shape[:2]

# Clip so sunglasses don't exceed boundaries
end_x = min(x + target_w, w)
end_y = min(y + target_h, h)

# Adjust width/height based on clipping
overlay_w = end_x - x
overlay_h = end_y - y

# Resize glasses to fit adjusted size
glassBGR_resized = cv2.resize(glassBGR, (overlay_w, overlay_h))
glassAlpha_resized = cv2.resize(glassPNG[..., 3] / 255.0, (overlay_w, overlay_h))

# Extract ROI of face
roi = faceWithGlasses[y:end_y, x:end_x]

# Alpha blending
for c in range(3):
    roi[..., c] = roi[..., c] * (1 - glassAlpha_resized) + glassBGR_resized[..., c] * glassAlpha_resized

# Put ROI back into image
faceWithGlasses[y:end_y, x:end_x] = roi

# Plot
plt.figure(figsize=[15, 8])
plt.subplot(1, 2, 1)
plt.imshow(faceImage[..., ::-1])
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(faceWithGlasses[..., ::-1])
plt.title("Face with Sunglasses")
plt.axis("off")

plt.tight_layout()
plt.show()
```

![WhatsApp Image 2025-09-24 at 11 20 10_59856039](https://github.com/user-attachments/assets/a47f86f5-c857-46d1-9db5-23162b1f8a02)

Feel free to fork, contribute, or customize this project for your creative needs!

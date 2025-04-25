import numpy as np
import cv2
from tqdm import tqdm

data = np.load("./masks_dance2.npy")
print(data.shape, data.dtype)
for mask in tqdm(data):
    image = np.zeros_like(mask, dtype=np.uint8)
    image[mask] = 255
    cv2.imshow("data", image)
    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()
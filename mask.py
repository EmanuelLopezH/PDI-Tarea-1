import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

input_folder = 'output'
output_folder = 'output/results'
os.makedirs(output_folder, exist_ok=True)

lower = np.array([74, 140, 110])
upper = np.array([76, 195, 250])

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        image = cv.imread(image_path)
        image = cv.resize(image, (240, 420))
        image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        mask = cv.inRange(image_hsv, lower - 255, upper)
        img = cv.dilate(mask, None, iterations=5)
        img = cv.erode(img, None, iterations=5)

        # Guardar resultados
        cv.imwrite(os.path.join(output_folder, f'processed_{filename}'), img)

        # Mostrar resultados (opcional)
        # plt.figure(figsize=(10, 3))
        # plt.subplot(1, 3, 1)
        # plt.title('Original')
        # plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.subplot(1, 3, 2)
        # plt.title('Mask')
        # plt.imshow(mask, cmap='gray')
        # plt.axis('off')
        # plt.subplot(1, 3, 3)
        # plt.title('Mask Applied')
        # plt.imshow(img, cmap='gray')
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()

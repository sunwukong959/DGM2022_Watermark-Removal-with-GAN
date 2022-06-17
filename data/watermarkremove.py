#!/usr/bin/env python3
import os

import cv2
import numpy as np


def xray_autocrop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [cv2.boundingRect(c) for c in contours]
    areas = [b[2] * b[3] for b in bboxes]
    bbox = bboxes[np.argmax(areas)]
    x, y, w, h = bbox
    return gray[y:y + h, x:x + w]


def xray_remove_watermark(image_A, image_B):
    image_A = autocrop(image_A)
    image_B = autocrop(image_B)
    image_B_flip = cv2.flip(image_B, 1)
    h, w = image_B_flip.shape
    image_AB = np.zeros((h, w), np.uint8)
    image_AB[:, :w // 2] = image_A[:, :w // 2]
    image_AB[:, w // 2:] = image_B_flip[:, w // 2:]
    return image_AB


def main():
    previous = ''
    os.makedirs('no_watermark', exist_ok=True)
    for i, filename in enumerate(sorted(os.listdir('raw'))):
        if i % 2 == 0:
            previous = filename
            continue
        image_A = cv2.imread(os.path.join('raw', previous))
        image_B = cv2.imread(os.path.join('raw', filename))
        image_AB = remove_watermark(image_A, image_B)
        cv2.imwrite(f'no_watermark/image_{i // 2:04d}.png', image_AB)


if __name__ == '__main__':
    main()

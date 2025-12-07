#!/usr/bin/env python3
"""
Rozszerzenie datasetu YOLO przez augmentację.
Generuje dodatkowe warianty zdjęć z transformacjami.
"""

import os
import cv2
import numpy as np
import glob
import random
from pathlib import Path

# Parametry
YOLO_DATASET_DIR = 'yolo_dataset'
AUGMENTATIONS_PER_IMAGE = 2  # Ile wersji każdego obrazu wygenerować
SEED = 2137

random.seed(SEED)
np.random.seed(SEED)


def load_yolo_labels(label_path):
    """Wczytaj etykiety YOLO z pliku."""
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    boxes.append([class_id, x_center, y_center, width, height])
    return boxes


def save_yolo_labels(label_path, boxes):
    """Zapisz etykiety YOLO do pliku."""
    with open(label_path, 'w') as f:
        for box in boxes:
            class_id = int(box[0])
            x_center, y_center, width, height = box[1:5]
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def augment_image_and_boxes(image, boxes):
    """
    Aplikuje losowe transformacje do obrazu i odpowiednio modyfikuje bounding boxy.
    Zwraca (augmented_image, augmented_boxes).
    """
    h, w = image.shape[:2]
    aug_image = image.copy()
    aug_boxes = [box[:] for box in boxes]  # Kopia
    
    # 1. Horizontal flip (30% szans - mniejszy nacisk)
    if random.random() > 0.7:
        aug_image = cv2.flip(aug_image, 1)
        for box in aug_boxes:
            box[1] = 1.0 - box[1]  # x_center = 1 - x_center
    
    # 2. Brightness & Contrast (20% szans - mniejszy nacisk)
    if random.random() > 0.8:
        alpha = random.uniform(0.8, 1.2)  # Mniejszy zakres kontrastu
        beta = random.randint(-20, 20)     # Mniejszy zakres jasności
        aug_image = cv2.convertScaleAbs(aug_image, alpha=alpha, beta=beta)
    
    # 3. Gaussian Noise (10% szans - minimalny nacisk)
    if random.random() > 0.9:
        noise = np.random.normal(0, 10, aug_image.shape).astype(np.uint8)
        aug_image = cv2.add(aug_image, noise)
    
    # 4. Gaussian Blur (10% szans - minimalny nacisk)
    if random.random() > 0.9:
        kernel_size = random.choice([3, 5])
        aug_image = cv2.GaussianBlur(aug_image, (kernel_size, kernel_size), 0)
    
    # 5. Rotation (90% szans - GŁÓWNY nacisk, większy zakres kątów)
    if random.random() > 0.1:
        angle = random.uniform(-30, 30)  # Zwiększony zakres rotacji
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aug_image = cv2.warpAffine(aug_image, M, (w, h), 
                                   borderMode=cv2.BORDER_REFLECT)
        
        # Rotacja bboxów (przybliżenie - zachowujemy prostokąty po rotacji)
        for box in aug_boxes:
            x_c, y_c = box[1] * w, box[2] * h
            box_w, box_h = box[3] * w, box[4] * h
            
            # Rotuj środek
            point = np.array([x_c, y_c, 1.0])
            rotated = M @ point
            
            # Po rotacji bbox może być większy - zwiększamy trochę
            scale_factor = 1.0 + abs(angle) / 90.0 * 0.3
            new_w = min(box_w * scale_factor, w)
            new_h = min(box_h * scale_factor, h)
            
            # Normalizuj z powrotem
            box[1] = max(0.0, min(1.0, rotated[0] / w))
            box[2] = max(0.0, min(1.0, rotated[1] / h))
            box[3] = max(0.01, min(1.0, new_w / w))
            box[4] = max(0.01, min(1.0, new_h / h))
    
    # 6. Shift (60% szans - umiarkowany nacisk)
    if random.random() > 0.4:
        shift_x = random.randint(-int(w * 0.15), int(w * 0.15))  # Większy zakres
        shift_y = random.randint(-int(h * 0.15), int(h * 0.15))
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        aug_image = cv2.warpAffine(aug_image, M, (w, h),
                                   borderMode=cv2.BORDER_REFLECT)
        
        for box in aug_boxes:
            box[1] = max(0.0, min(1.0, box[1] + shift_x / w))
            box[2] = max(0.0, min(1.0, box[2] + shift_y / h))
    
    # 7. Scale (70% szans - zwiększony nacisk)
    if random.random() > 0.3:
        scale = random.uniform(0.7, 1.3)  # Większy zakres skalowania
        new_w, new_h = int(w * scale), int(h * scale)
        aug_image = cv2.resize(aug_image, (new_w, new_h))
        
        # Crop/pad do oryginalnego rozmiaru
        if scale > 1.0:  # Crop
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            aug_image = aug_image[start_y:start_y+h, start_x:start_x+w]
            
            # Adjust boxes
            for box in aug_boxes:
                box[1] = (box[1] * scale - start_x / w)
                box[2] = (box[2] * scale - start_y / h)
                box[3] = box[3] * scale
                box[4] = box[4] * scale
        else:  # Pad
            pad_x = (w - new_w) // 2
            pad_y = (h - new_h) // 2
            aug_image = cv2.copyMakeBorder(aug_image, pad_y, h-new_h-pad_y, 
                                          pad_x, w-new_w-pad_x, 
                                          cv2.BORDER_REFLECT)
            
            for box in aug_boxes:
                box[1] = (box[1] * new_w + pad_x) / w
                box[2] = (box[2] * new_h + pad_y) / h
                box[3] = box[3] * scale
                box[4] = box[4] * scale
        
        # Clamp boxes
        for box in aug_boxes:
            box[1] = max(0.0, min(1.0, box[1]))
            box[2] = max(0.0, min(1.0, box[2]))
            box[3] = max(0.01, min(1.0, box[3]))
            box[4] = max(0.01, min(1.0, box[4]))
    
    # Filtruj boxy poza obrazem
    valid_boxes = []
    for box in aug_boxes:
        x_c, y_c, bw, bh = box[1:5]
        # Sprawdź czy bbox jest przynajmniej częściowo w obrazie
        if (0 <= x_c <= 1 and 0 <= y_c <= 1 and 
            bw > 0.01 and bh > 0.01 and bw <= 1 and bh <= 1):
            valid_boxes.append(box)
    
    return aug_image, valid_boxes


def augment_split(split_name):
    """Augmentuj dany split (train lub val)."""
    img_dir = os.path.join(YOLO_DATASET_DIR, 'images', split_name)
    lbl_dir = os.path.join(YOLO_DATASET_DIR, 'labels', split_name)
    
    if not os.path.exists(img_dir):
        print(f"Brak katalogu {img_dir}")
        return
    
    # Znajdź wszystkie obrazy
    image_files = []
    for ext in ['*.bmp', '*.jpg', '*.png', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(img_dir, ext)))
    
    if not image_files:
        print(f"Brak obrazów w {img_dir}")
        return
    
    print(f"\n{split_name.upper()}: Augmentuję {len(image_files)} obrazów...")
    
    augmented_count = 0
    
    for img_path in image_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1]
        lbl_path = os.path.join(lbl_dir, basename + '.txt')
        
        # Wczytaj obraz i etykiety
        image = cv2.imread(img_path)
        if image is None:
            print(f"Nie można wczytać {img_path}")
            continue
        
        boxes = load_yolo_labels(lbl_path)
        if not boxes:
            print(f"Brak etykiet dla {basename}, pomijam")
            continue
        
        # Generuj augmentacje
        for i in range(AUGMENTATIONS_PER_IMAGE):
            aug_img, aug_boxes = augment_image_and_boxes(image, boxes)
            
            if not aug_boxes:
                continue  # Pomijamy jeśli wszystkie boxy wyszły poza obraz
            
            # Zapisz augmentowany obraz i etykiety
            aug_basename = f"{basename}_aug{i+1}"
            aug_img_path = os.path.join(img_dir, aug_basename + ext)
            aug_lbl_path = os.path.join(lbl_dir, aug_basename + '.txt')
            
            cv2.imwrite(aug_img_path, aug_img)
            save_yolo_labels(aug_lbl_path, aug_boxes)
            augmented_count += 1
    
    print(f"  ✓ Dodano {augmented_count} augmentowanych obrazów")
    
    # Policz końcową liczbę
    total_images = len(glob.glob(os.path.join(img_dir, '*.*')))
    print(f"  Razem w {split_name}: {total_images} obrazów")


def main():
    """Główna funkcja."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    if not os.path.exists(YOLO_DATASET_DIR):
        print(f"Brak katalogu {YOLO_DATASET_DIR}. Uruchom najpierw labelme_to_yolo.py")
        return
    
    print("=" * 60)
    print("AUGMENTACJA DATASETU YOLO")
    print("=" * 60)
    print(f"Katalog: {YOLO_DATASET_DIR}")
    print(f"Augmentacji na obraz: {AUGMENTATIONS_PER_IMAGE}")
    
    # Augmentuj train i val
    augment_split('train')
    augment_split('val')
    
    print("\n" + "=" * 60)
    print("✓ AUGMENTACJA ZAKOŃCZONA")
    print("=" * 60)
    print("\nDataset jest gotowy do treningu!")
    print(f"Użyj: yolo task=detect mode=train model=yolov8n.pt \\")
    print(f"       data={YOLO_DATASET_DIR}/data.yaml epochs=100 imgsz=640")


if __name__ == '__main__':
    main()

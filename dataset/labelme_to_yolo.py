#!/usr/bin/env python3
"""
Konwersja anotacji LabelMe (JSON) na format YOLO.
Przetwarza tylko zdjęcia z labelami, pomija obrazy bez adnotacji.
"""

import os
import json
import glob
import shutil
from pathlib import Path
from PIL import Image

# Ścieżki
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, 'images')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'yolo_dataset')
TRAIN_RATIO = 0.8  # 80% train, 20% val

# Mapowanie klas (dodaj więcej jeśli potrzeba)
CLASS_MAPPING = {
    'bottle': 0,
    #'bottle2': 1, #jednak gaśnica xd
    'box': 1,
    'crate': 2,
    'crowbar': 5,
    #'evilsquare': 5,
    'grenade': 3,
    'scissors': 4,
}


def convert_labelme_to_yolo(json_path):
    """Konwertuje pojedynczy plik JSON z LabelMe na format YOLO."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Pobierz wymiary obrazu
    img_width = data.get('imageWidth')
    img_height = data.get('imageHeight')
    
    if not img_width or not img_height:
        # Spróbuj wczytać z obrazu
        img_path = data.get('imagePath', '')
        if img_path:
            # imagePath może być relatywną ścieżką, znajdź faktyczny obraz
            json_dir = os.path.dirname(json_path)
            json_base = os.path.splitext(os.path.basename(json_path))[0]
            
            # Szukaj obrazu z tym samym basename
            for ext in ['.bmp', '.jpg', '.png', '.jpeg']:
                potential_img = os.path.join(json_dir, json_base + ext)
                if os.path.exists(potential_img):
                    try:
                        with Image.open(potential_img) as img:
                            img_width, img_height = img.size
                        break
                    except Exception as e:
                        print(f"Błąd odczytu obrazu {potential_img}: {e}")
                        continue
    
    if not img_width or not img_height:
        print(f"Nie można określić wymiarów dla {json_path}")
        return None, None
    
    # Przetwórz wszystkie shape'y
    yolo_lines = []
    shapes = data.get('shapes', [])
    
    if not shapes:
        return None, None  # Brak anotacji
    
    for shape in shapes:
        label = shape.get('label', '').lower()
        
        # Mapuj label na class_id
        if label not in CLASS_MAPPING:
            print(f"Nieznana klasa '{label}' w {json_path}, pomijam")
            continue
        
        class_id = CLASS_MAPPING[label]
        points = shape.get('points', [])
        
        if not points:
            continue
        
        # Oblicz bounding box z punktów (dla rectangle lub polygon)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        
        # Konwersja do formatu YOLO (znormalizowane środek + szerokość/wysokość)
        x_center = (x_min + x_max) / 2.0 / img_width
        y_center = (y_min + y_max) / 2.0 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        # Sprawdź poprawność wartości
        if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1:
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        else:
            print(f"Nieprawidłowe wartości bbox w {json_path}: {x_center}, {y_center}, {width}, {height}")
    
    return yolo_lines, (img_width, img_height)


def main():
    """Główna funkcja konwertująca cały dataset."""
    
    # Utwórz strukturę katalogów YOLO
    train_img_dir = os.path.join(OUTPUT_DIR, 'images', 'train')
    val_img_dir = os.path.join(OUTPUT_DIR, 'images', 'val')
    train_lbl_dir = os.path.join(OUTPUT_DIR, 'labels', 'train')
    val_lbl_dir = os.path.join(OUTPUT_DIR, 'labels', 'val')
    
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Znajdź wszystkie pliki JSON
    json_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.json')))
    
    if not json_files:
        print(f"Nie znaleziono plików JSON w {INPUT_DIR}")
        return
    
    print(f"Znaleziono {len(json_files)} plików JSON")
    
    processed = 0
    skipped = 0
    
    # Przetwórz każdy plik JSON
    valid_samples = []
    
    for json_path in json_files:
        yolo_lines, img_dims = convert_labelme_to_yolo(json_path)
        
        if not yolo_lines:
            skipped += 1
            continue
        
        # Znajdź odpowiedni plik obrazu
        json_base = os.path.splitext(os.path.basename(json_path))[0]
        img_path = None
        
        for ext in ['.bmp', '.jpg', '.png', '.jpeg']:
            potential_img = os.path.join(INPUT_DIR, json_base + ext)
            if os.path.exists(potential_img):
                img_path = potential_img
                break
        
        if not img_path:
            print(f"Nie znaleziono obrazu dla {json_path}")
            skipped += 1
            continue
        
        valid_samples.append({
            'img_path': img_path,
            'yolo_lines': yolo_lines,
            'basename': json_base,
            'ext': os.path.splitext(img_path)[1]
        })
        
        processed += 1
    
    # Podziel na train/val
    import random
    random.seed(42)
    random.shuffle(valid_samples)
    
    split_idx = int(len(valid_samples) * TRAIN_RATIO)
    train_samples = valid_samples[:split_idx]
    val_samples = valid_samples[split_idx:]
    
    print(f"\nPodział: {len(train_samples)} train, {len(val_samples)} val")
    
    # Kopiuj pliki do odpowiednich katalogów
    for samples, img_dir, lbl_dir, split_name in [
        (train_samples, train_img_dir, train_lbl_dir, 'train'),
        (val_samples, val_img_dir, val_lbl_dir, 'val')
    ]:
        for sample in samples:
            # Kopiuj obraz
            dst_img = os.path.join(img_dir, sample['basename'] + sample['ext'])
            shutil.copy2(sample['img_path'], dst_img)
            
            # Zapisz etykiety YOLO
            dst_lbl = os.path.join(lbl_dir, sample['basename'] + '.txt')
            with open(dst_lbl, 'w', encoding='utf-8') as f:
                f.write('\n'.join(sample['yolo_lines']))
    
    # Utwórz plik classes.txt
    classes_file = os.path.join(OUTPUT_DIR, 'classes.txt')
    sorted_classes = sorted(CLASS_MAPPING.items(), key=lambda x: x[1])
    with open(classes_file, 'w', encoding='utf-8') as f:
        for class_name, _ in sorted_classes:
            f.write(f"{class_name}\n")
    
    # Utwórz plik konfiguracyjny YAML dla YOLO
    yaml_content = f"""# Dataset configuration for YOLO
path: {os.path.abspath(OUTPUT_DIR)}
train: images/train
val: images/val

# Number of classes
nc: {len(CLASS_MAPPING)}

# Class names
names: {[name for name, _ in sorted_classes]}
"""
    
    yaml_file = os.path.join(OUTPUT_DIR, 'data.yaml')
    with open(yaml_file, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"\n✓ Konwersja zakończona!")
    print(f"  Przetworzone: {processed}")
    print(f"  Pominięte (brak anotacji): {skipped}")
    print(f"\nWygenerowane pliki:")
    print(f"  Dataset: {OUTPUT_DIR}/")
    print(f"  Config: {yaml_file}")
    print(f"  Classes: {classes_file}")
    print(f"\nAby trenować YOLO, użyj:")
    print(f"  yolo task=detect mode=train model=yolov8n.pt data={yaml_file} epochs=100 imgsz=640")


if __name__ == '__main__':
    main()

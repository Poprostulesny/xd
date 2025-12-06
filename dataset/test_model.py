#!/usr/bin/env python3
"""Test wytrenowanego modelu YOLO na nowych obrazach."""

from ultralytics import YOLO
import cv2
import os
import glob

def test_on_images(model_path, test_images_dir, output_dir='test_results'):
    """Testuj model na obrazach z katalogu."""
    
    # Załaduj wytrenowany model
    model = YOLO(model_path)
    
    # Utwórz katalog wynikowy
    os.makedirs(output_dir, exist_ok=True)
    
    # Znajdź wszystkie obrazy
    image_files = []
    for ext in ['*.bmp', '*.jpg', '*.png', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(test_images_dir, ext)))
    
    if not image_files:
        print(f"Brak obrazów w {test_images_dir}")
        return
    
    print(f"Znaleziono {len(image_files)} obrazów do testu")
    print("=" * 60)
    
    for img_path in image_files[:10]:  # Testuj pierwsze 10
        print(f"\nPrzetwarzam: {os.path.basename(img_path)}")
        
        # Detekcja
        results = model.predict(
            source=img_path,
            conf=0.25,  # Próg pewności
            save=False,
            verbose=False
        )
        
        # Rysuj wyniki
        img = cv2.imread(img_path)
        result = results[0]
        
        # Informacje o detekcjach
        boxes = result.boxes
        if len(boxes) > 0:
            print(f"  Wykryto {len(boxes)} obiektów:")
            for i, box in enumerate(boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                class_name = model.names[cls]
                print(f"    {i+1}. {class_name}: {conf:.2f}")
                
                # Rysuj bbox
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), 
                             (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(img, label, (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print("  Brak detekcji")
        
        # Zapisz wynik
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, img)
    
    print("\n" + "=" * 60)
    print(f"✓ Wyniki zapisane w: {output_dir}/")
    print(f"Otwórz katalog aby zobaczyć obrazy z detekcjami")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Ścieżka do modelu
    model_path = 'runs/crate_detection/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"Brak modelu w {model_path}")
        print("Uruchom najpierw train_yolo.py")
        return
    
    print("=" * 60)
    print("TEST MODELU YOLO")
    print("=" * 60)
    print(f"Model: {model_path}\n")
    
    # Test na zbiorze walidacyjnym
    print("\n1. Test na zbiorze walidacyjnym:")
    test_on_images(model_path, 
                   'yolo_dataset/images/val', 
                   'test_results/val')
    
    # Test na nowych obrazach (jeśli są)
    if os.path.exists('../data_oriented/brudne'):
        print("\n2. Test na nowych obrazach (data_oriented/brudne):")
        test_on_images(model_path,
                       '../data_oriented/brudne',
                       'test_results/new_images')
    
    # Pokaż metryki
    print("\n" + "=" * 60)
    print("METRYKI MODELU")
    print("=" * 60)
    model = YOLO(model_path)
    metrics = model.val(data='yolo_dataset/data.yaml')
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall: {metrics.box.mr:.3f}")


if __name__ == '__main__':
    main()

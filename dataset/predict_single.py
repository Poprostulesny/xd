#!/usr/bin/env python3
"""Test modelu YOLO na pojedynczym zdjęciu."""

from ultralytics import YOLO
import cv2
import sys
import os

def predict_image(model_path, image_path, output_path='result.jpg', conf_threshold=0.25):
    """Wykryj obiekty na pojedynczym zdjęciu."""
    
    # Załaduj model
    if not os.path.exists(model_path):
        print(f"Błąd: Brak modelu w {model_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"Błąd: Brak zdjęcia {image_path}")
        return
    
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Zdjęcie: {image_path}")
    print(f"Próg pewności: {conf_threshold}")
    print("=" * 60)
    
    model = YOLO(model_path)
    
    # Detekcja
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=False,
        verbose=False
    )
    
    # Wczytaj obraz
    img = cv2.imread(image_path)
    
    # Konwertuj grayscale na BGR jeśli potrzeba
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    result = results[0]
    boxes = result.boxes
    
    print(f"\nWykryto {len(boxes)} obiektów:\n")
    
    # Rysuj każdy bbox
    for i, box in enumerate(boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        class_name = model.names[cls]
        print(f"  {i+1}. {class_name}: {conf:.2%} pewności")
        print(f"     Pozycja: ({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)})")
        
        # Rysuj prostokąt
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), 
                     (0, 255, 0), 3)
        
        # Etykieta
        label = f"{class_name} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Tło dla tekstu
        cv2.rectangle(img, (int(x1), int(y1)-label_size[1]-10), 
                     (int(x1)+label_size[0], int(y1)), (0, 255, 0), -1)
        
        # Tekst
        cv2.putText(img, label, (int(x1), int(y1)-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    if len(boxes) == 0:
        print("  Brak detekcji")
    
    # Zapisz wynik
    cv2.imwrite(output_path, img)
    print(f"\n✓ Wynik zapisany: {output_path}")
    
    # Wyświetl obraz - przeskaluj jeśli za duży
    h, w = img.shape[:2]
    max_dim = 1200
    if h > max_dim or w > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        display_img = cv2.resize(img, (new_w, new_h))
    else:
        display_img = img
    
    cv2.imshow('Detekcja - naciśnij dowolny klawisz aby zamknąć', display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("=" * 60)


def main():
    
    image_path = sys.argv[1]
    conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    #image_path = "dataset/yolo_dataset/images/val/48001F003202511190110 czarno_aug1.bmp"
    #conf = 0.5
    
    # Domyślna ścieżka do modelu
    model_path = 'runs/detect/train8/weights/best.pt'
    
    # Nazwa pliku wyjściowego
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"result_{basename}.jpg"
    
    predict_image(model_path, image_path, output_path, conf)


if __name__ == '__main__':
    main()

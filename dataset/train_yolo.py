#!/usr/bin/env python3
"""Skrypt treningowy YOLOv8 dla detekcji obiektów."""

from ultralytics import YOLO
import os

def main():
    # Zmień katalog roboczy
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("=" * 60)
    print("TRENING YOLOV8")
    print("=" * 60)
    
    # Załaduj model YOLOv8 nano (pretrained)
    model = YOLO('yolov8n.pt')
    
    # Trenuj model
    results = model.train(
        data='yolo_dataset/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        patience=20,
        project='runs',
        name='crate_detection',
        save=True,
        plots=True,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("✓ TRENING ZAKOŃCZONY")
    print("=" * 60)
    print(f"Model zapisany w: runs/crate_detection/weights/best.pt")
    print(f"Wyniki w: runs/crate_detection/")
    
    # Walidacja
    print("\nUruchamiam walidację...")
    metrics = model.val()
    
    print(f"\nMetryki:")
    print(f"  mAP50: {metrics.box.map50:.3f}")
    print(f"  mAP50-95: {metrics.box.map:.3f}")
    
    return results

if __name__ == '__main__':
    main()

import cv2
import numpy as np


def crop_white_background(image_path, threshold=240, padding=5, background_ratio=0.95, edge_scan_depth=10):
    """
    Przycina obraz w skali szarości usuwając białe tło.
    
    Args:
        image_path: Ścieżka do obrazu lub sam obraz (numpy array)
        threshold: Próg jasności - piksele jaśniejsze niż ta wartość są uważane za tło (0-255)
        padding: Dodatkowy margines wokół zawartości (w pikselach)
        background_ratio: Jaki procent pikseli w rzędzie/kolumnie musi być tłem, aby uznać ją za pustą (0-1)
        edge_scan_depth: Ile kolejnych rzędów/kolumn od krawędzi musi być tłem, aby uznać je za puste
    
    Returns:
        Przycięty obraz (numpy array)
    """
    # Wczytaj obraz jeśli podano ścieżkę
    if isinstance(image_path, str):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Nie można wczytać obrazu: {image_path}")
    else:
        img = image_path
    
    h, w = img.shape
    
    # Znajdź maski tła
    background_mask = img >= threshold
    
    # Sprawdź każdy rząd - ile procent to tło
    row_background_ratio = background_mask.mean(axis=1)
    
    # Sprawdź każdą kolumnę - ile procent to tło
    col_background_ratio = background_mask.mean(axis=0)
    
    # Znajdź pierwszy rząd z góry, od którego zaczyna się zawartość
    # Ignorujemy pojedyncze odstające piksele przy krawędziach
    y_min = 0
    i = 0
    while i < h - edge_scan_depth:
        # Sprawdź następne edge_scan_depth rzędów
        next_rows = row_background_ratio[i:i+edge_scan_depth]
        # Jeśli większość (co najmniej 70%) to tło, uznaj za tło
        if np.sum(next_rows >= background_ratio) >= edge_scan_depth * 0.7:
            y_min = i + edge_scan_depth
            i += edge_scan_depth
        else:
            break
    
    # Znajdź ostatni rząd od dołu
    y_max = h
    i = h - 1
    while i >= edge_scan_depth:
        start_idx = max(0, i - edge_scan_depth + 1)
        prev_rows = row_background_ratio[start_idx:i+1]
        if np.sum(prev_rows >= background_ratio) >= edge_scan_depth * 0.7:
            y_max = start_idx
            i -= edge_scan_depth
        else:
            break
    
    # Znajdź pierwszą kolumnę z lewej
    x_min = 0
    i = 0
    while i < w - edge_scan_depth:
        next_cols = col_background_ratio[i:i+edge_scan_depth]
        if np.sum(next_cols >= background_ratio) >= edge_scan_depth * 0.7:
            x_min = i + edge_scan_depth
            i += edge_scan_depth
        else:
            break
    
    # Znajdź ostatnią kolumnę z prawej
    x_max = w
    i = w - 1
    while i >= edge_scan_depth:
        start_idx = max(0, i - edge_scan_depth + 1)
        prev_cols = col_background_ratio[start_idx:i+1]
        if np.sum(prev_cols >= background_ratio) >= edge_scan_depth * 0.7:
            x_max = start_idx
            i -= edge_scan_depth
        else:
            break
    
    # Dodaj padding
    y_min = max(0, y_min - padding)
    y_max = min(h, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(w, x_max + padding)
    
    # Przytnij obraz
    cropped = img[y_min:y_max, x_min:x_max]
    
    return cropped


def crop_and_save(input_path, output_path, threshold=240, padding=5, background_ratio=0.95, edge_scan_depth=10):
    """
    Przycina obraz i zapisuje wynik do pliku.
    
    Args:
        input_path: Ścieżka do obrazu wejściowego
        output_path: Ścieżka zapisu przyciętego obrazu
        threshold: Próg jasności dla tła
        padding: Margines wokół zawartości
        background_ratio: Jaki procent pikseli w rzędzie/kolumnie musi być tłem
        edge_scan_depth: Głębokość skanowania od krawędzi
    """
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Nie można wczytać obrazu: {input_path}")
    
    cropped = crop_white_background(img, threshold, padding, background_ratio, edge_scan_depth)
    cv2.imwrite(output_path, cropped)
    
    return cropped


import os
from pathlib import Path


def process_directory(input_dir, output_dir, threshold=240, padding=5, background_ratio=0.95, edge_scan_depth=10):
    """
    Przetwarza wszystkie obrazy BMP w katalogu i zapisuje przycięte wersje.
    
    Args:
        input_dir: Ścieżka do katalogu wejściowego
        output_dir: Ścieżka do katalogu wyjściowego
        threshold: Próg jasności dla tła
        padding: Margines wokół zawartości
        background_ratio: Jaki procent pikseli w rzędzie/kolumnie musi być tłem
        edge_scan_depth: Głębokość skanowania od krawędzi
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Utwórz katalog wyjściowy jeśli nie istnieje
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Znajdź wszystkie pliki BMP
    bmp_files = list(input_path.glob("*.bmp"))
    
    if not bmp_files:
        print(f"Nie znaleziono plików BMP w {input_dir}")
        return
    
    print(f"Znaleziono {len(bmp_files)} plików BMP w {input_dir}")
    
    success_count = 0
    error_count = 0
    
    for i, input_file in enumerate(bmp_files, 1):
        output_file = output_path / input_file.name
        
        try:
            result = crop_and_save(
                str(input_file), 
                str(output_file), 
                threshold=threshold, 
                padding=padding, 
                background_ratio=background_ratio,
                edge_scan_depth=edge_scan_depth
            )
            
            original_shape = cv2.imread(str(input_file), cv2.IMREAD_GRAYSCALE).shape
            print(f"[{i}/{len(bmp_files)}] {input_file.name}: {original_shape} → {result.shape}")
            success_count += 1
            
        except Exception as e:
            print(f"[{i}/{len(bmp_files)}] Błąd przy {input_file.name}: {e}")
            error_count += 1
    
    print(f"\nPrzetworzono: {success_count} plików")
    if error_count > 0:
        print(f"Błędów: {error_count}")


# Przykład użycia
if __name__ == "__main__":
    # Przetwórz czyste
    print("=" * 60)
    print("Przetwarzanie data/czyste...")
    print("=" * 60)
    process_directory("data/czyste", "data_cropped/czyste")
    
    # Przetwórz brudne
    print("\n" + "=" * 60)
    print("Przetwarzanie data/brudne...")
    print("=" * 60)
    process_directory("data/brudne", "data_cropped/brudne")

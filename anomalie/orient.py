import cv2
import numpy as np
from pathlib import Path


def detect_car_orientation(image_path):
    """
    Wykrywa orientację samochodu na podstawie rozkładu ciemnych pikseli.
    Zakłada, że przód samochodu (koła, silnik) jest ciemniejszy niż tył.
    
    Args:
        image_path: Ścieżka do obrazu lub sam obraz (numpy array)
    
    Returns:
        'left' jeśli przód jest po lewej, 'right' jeśli po prawej
    """
    # Wczytaj obraz jeśli podano ścieżkę
    if isinstance(image_path, str):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Nie można wczytać obrazu: {image_path}")
    else:
        img = image_path
    
    h, w = img.shape
    
    # Podziel obraz na lewą i prawą połowę
    left_half = img[:, :w//2]
    right_half = img[:, w//2:]
    
    # Oblicz średnią jasność dla każdej połowy
    # Niższa wartość = ciemniejsze = więcej zawartości (koła, silnik)
    left_mean = left_half.mean()
    right_mean = right_half.mean()
    
    # Alternatywnie: policz ciemne piksele (< 200)
    left_dark_pixels = np.sum(left_half < 200)
    right_dark_pixels = np.sum(right_half < 200)
    
    # Przód jest tam gdzie więcej ciemnych pikseli
    if left_dark_pixels > right_dark_pixels:
        return 'left'
    else:
        return 'right'


def orient_car(image_path, target_orientation='right'):
    """
    Orientuje obraz samochodu tak, aby przód był po wskazanej stronie.
    
    Args:
        image_path: Ścieżka do obrazu lub sam obraz (numpy array)
        target_orientation: 'left' lub 'right' - gdzie ma być przód
    
    Returns:
        Obraz z odpowiednią orientacją (numpy array)
    """
    # Wczytaj obraz jeśli podano ścieżkę
    if isinstance(image_path, str):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Nie można wczytać obrazu: {image_path}")
    else:
        img = image_path
    
    # Wykryj aktualną orientację
    current_orientation = detect_car_orientation(img)
    
    # Jeśli trzeba, odwróć obraz
    if current_orientation != target_orientation:
        img = cv2.flip(img, 1)  # flip horizontally (1 = horizontal, 0 = vertical)
        return img, True  # True = flipped
    
    return img, False  # False = not flipped


def orient_and_save(input_path, output_path, target_orientation='right'):
    """
    Orientuje obraz i zapisuje wynik do pliku.
    
    Args:
        input_path: Ścieżka do obrazu wejściowego
        output_path: Ścieżka zapisu zorientowanego obrazu
        target_orientation: 'left' lub 'right' - gdzie ma być przód
    
    Returns:
        (oriented_image, was_flipped)
    """
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Nie można wczytać obrazu: {input_path}")
    
    oriented, flipped = orient_car(img, target_orientation)
    cv2.imwrite(output_path, oriented)
    
    return oriented, flipped


def process_directory(input_dir, output_dir, target_orientation='right'):
    """
    Przetwarza wszystkie obrazy BMP w katalogu i orientuje je.
    
    Args:
        input_dir: Ścieżka do katalogu wejściowego
        output_dir: Ścieżka do katalogu wyjściowego
        target_orientation: 'left' lub 'right' - gdzie ma być przód
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
    flipped_count = 0
    error_count = 0
    
    for i, input_file in enumerate(bmp_files, 1):
        output_file = output_path / input_file.name
        
        try:
            oriented, flipped = orient_and_save(
                str(input_file), 
                str(output_file), 
                target_orientation=target_orientation
            )
            
            status = "odwrócono" if flipped else "bez zmian"
            print(f"[{i}/{len(bmp_files)}] {input_file.name}: {status}")
            success_count += 1
            if flipped:
                flipped_count += 1
            
        except Exception as e:
            print(f"[{i}/{len(bmp_files)}] Błąd przy {input_file.name}: {e}")
            error_count += 1
    
    print(f"\nPrzetworzono: {success_count} plików")
    print(f"Odwrócono: {flipped_count} plików")
    if error_count > 0:
        print(f"Błędów: {error_count}")


# Przykład użycia
if __name__ == "__main__":
    # Przetwórz przycięte obrazy - czyste
    print("=" * 60)
    print("Orientowanie data_cropped/czyste...")
    print("=" * 60)
    process_directory("data_cropped/czyste", "data_oriented/czyste", target_orientation='right')
    
    # Przetwórz przycięte obrazy - brudne
    print("\n" + "=" * 60)
    print("Orientowanie data_cropped/brudne...")
    print("=" * 60)
    process_directory("data_cropped/brudne", "data_oriented/brudne", target_orientation='right')

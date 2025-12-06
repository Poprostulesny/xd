import os
import shutil
from pathlib import Path

# Katalog źródłowy i docelowy
source_dir = Path("NAUKA")
dest_dir = Path("NAUKA_filtered")

# Utwórz nowy katalog główny
dest_dir.mkdir(exist_ok=True)

# Utwórz podfoldery dla "brudne" i "czyste"
(dest_dir / "brudne").mkdir(exist_ok=True)
(dest_dir / "czyste").mkdir(exist_ok=True)

# Przejdź przez wszystkie foldery w strukturze
for root, dirs, files in os.walk(source_dir):
    # Skopiuj tylko pliki Z "czarno" w nazwie
    for file in files:
        if "czarno" in file.lower() and file.endswith('.bmp'):
            src_file = Path(root) / file
            
            # Określ czy to "brudne" czy "czyste"
            if "brudne" in root:
                dst_file = dest_dir / "brudne" / file
            elif "czyste" in root:
                dst_file = dest_dir / "czyste" / file
            else:
                continue
                
            shutil.copy2(src_file, dst_file)
            print(f"Skopiowano: {src_file} -> {dst_file}")

print(f"\nGotowe! Nowa struktura została utworzona w folderze: {dest_dir}")

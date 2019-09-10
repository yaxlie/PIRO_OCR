import sys
import ocr

if len(sys.argv) > 1:
    result = ocr.ocr(str(sys.argv[1]))
    print('\n'.join(map(str, result)))
else:
    print("Podaj ścieżkę do zdjęcia.")

import sys
import ocr
import keras
import glob

MODEL_FILENAME = 'rec_digits.h5'


def main():
    if len(sys.argv) > 1:
        result = ocr.ocr(str(sys.argv[1]))
        print('\n'.join(map(str, result)))
    else:
        print("Nie podano ścieżki do pliku. Rozpoznawanie plików z katalogu /res")
        model = keras.models.load_model('saved_models/' + MODEL_FILENAME)

        for file in glob.iglob('res/**', recursive=True):
            print(file)
            if file.endswith('.jpg'):
                result = ocr.ocr(file, model)
                print('\n'.join(map(str, result)))


if __name__ == '__main__':
    main()

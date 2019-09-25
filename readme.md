

# Projekt 2: OCR - PiRO
Celem projektu było stworzenie programu analizującego zdjęcie przedstawiające ręcznie spisaną listę studentów ([lp.], imię, nazwisko, numer indeksu).

## Autorzy

| Imię | Nazwisko | Indeks |
| :- |:-:| -:|
| Bartosz | Nawrotek |  127297 |
| Marcin | Leśnierowski |  127301 |
  

### Praca została podzielona na 2 etapy:

1. Wykrywanie linii tekstu i przekazanie ich jako lista wycinków zdjęcia
2.  Rozpoznawanie numerów indeksu z pojedynczych linii
    

  
  
## 
* [Pełny opis wymagań](http://www.cs.put.poznan.pl/bwieloch/?page_id=872)
* [Repozytorium projektu](https://github.com/yaxlie/PIRO_OCR)
* [Notebook uczenia sieci](https://colab.research.google.com/drive/1QKe86ts1JJ1cyJLusWeOX3YBmSv395I_?fbclid=IwAR3uvwK6Rw3nrfj0-j9QK46lP5JW3kZWHrAMsqUahX-0579l9KxNvz2-W6c#scrollTo=qTXRvF3uTrzU](https://colab.research.google.com/drive/1QKe86ts1JJ1cyJLusWeOX3YBmSv395I_?fbclid=IwAR3uvwK6Rw3nrfj0-j9QK46lP5JW3kZWHrAMsqUahX-0579l9KxNvz2-W6c#scrollTo=qTXRvF3uTrzU))
  
  
  
  
  
  

## Wykrywanie linii

Zadaniem jest przetworzenie podanego zdjęcia i zwrócenie listy linii, które powinny zawierać [lp.], imię, nazwisko, numer indeksu.

  

### Kroki do rozwiązania problemu:

1.  Wykorzystanie skryptu `sliders.py` do wstępnego ustalenia parametrów, które pozwolą uwidocznić napisy, jednocześnie ukrywając szum.
(modyfikacja parametrów: *gamma, contrast, mean*)
2.  Wstępne przetworzenie zdjęcia - klasa `PreProcessing` w `processing.py`.
Użycie wcześniej dobranych parametrów i zastosowanie operacji morfologicznych, pozwala na uwidocznienie napisów, a następnie przekształcenie ich w kształty, które będzie można wykorzystać dalej.

| Oryginalne zdjęcie | Po processingu |
| - | - |
| <a target="_blank" href="https://github.com/yaxlie/PIRO_OCR/blob/master/res/img_1.jpg?raw=true"><img src="https://github.com/yaxlie/PIRO_OCR/blob/master/res/img_1.jpg?raw=true" width="80"></a> | <a target="_blank" href="https://github.com/yaxlie/PIRO_OCR/blob/feature/report/report/lines.png?raw=true"><img src="https://github.com/yaxlie/PIRO_OCR/blob/feature/report/report/lines.png?raw=true" width="80"></a> |
3. Przejście do klasy `LinesUtil`
4.  Znalezienie konturów, za pomocą funkcji `cv2.findContours`
5.  Stworzenie listy wycinków, za pomocą funkcji `cv2.convexHull`
6.  Analiza położenia poszczególnych wycinków. Wycinki, które znajdują się na podobnej wysokości, zostaną przydzielone do tej samej linii i zapisane na liście wynikowej.
7.  Zwrócenie listy linii. Każda linia zawiera wycinki, które niekoniecznie są poprawnym napisem. Dany wycinek może zawierać np. puste kratki. 
8. Kolejność elementów na liście odzwierciedla odwróconą kolejność na zdjęciu, tzn. elementy są wypisywane od dołu do góry oraz od prawej do lewej strony.

## Rozpoznawanie indeksów

1. Zaprojektowanie oraz przeprowadzenie procesu uczenia modelu. Sieć posiada 11 wyjść. Wyjścia o numerach 0 do 9 odpowiadają za klasyfikację cyfr, natomiast wyjście o numerze 10 jest odpowiedzialne za rozpoznawanie odstępów między cyframi. Proces uczenia został przeprowadzony na platformie colab. Kod wraz z opisem jest dostępny pod następującym adresem. [tutaj]([https://colab.research.google.com/drive/1QKe86ts1JJ1cyJLusWeOX3YBmSv395I_?fbclid=IwAR3uvwK6Rw3nrfj0-j9QK46lP5JW3kZWHrAMsqUahX-0579l9KxNvz2-W6c#scrollTo=ZJqm9adoY0WG](https://colab.research.google.com/drive/1QKe86ts1JJ1cyJLusWeOX3YBmSv395I_?fbclid=IwAR3uvwK6Rw3nrfj0-j9QK46lP5JW3kZWHrAMsqUahX-0579l9KxNvz2-W6c#scrollTo=ZJqm9adoY0WG))
2. Funkcja `get_index` w klasie `RecognizeNumbers` (`processing.py`).
3. Iterowanie po elementach przekazanej listy linii.
4. Dla każdej linii analizowany jest jej każdy element, który może być numerem indeksu.
5. Do wycinka dodawane są odstępy imitujące kratkę tak, aby napis nie znajdował się zbyt blisko krawędzi.
 
| Oryginalne zdjęcie | Po dodaniu paddingu |
| - | - |
| <a target="_blank" href="https://github.com/yaxlie/PIRO_OCR/blob/feature/report/report/cropped.PNG?raw=true"><img src="https://github.com/yaxlie/PIRO_OCR/blob/feature/report/report/cropped.PNG?raw=true" alt="cropped.jpg" width="80"></a> | <a target="_blank" href="https://github.com/yaxlie/PIRO_OCR/blob/feature/report/report/padded.PNG?raw=true"><img src="https://github.com/yaxlie/PIRO_OCR/blob/feature/report/report/padded.PNG?raw=true" width="80"></a> |
6. Próbkowanie wycinka (`sample_img`). Wycinane zostają okna o mniejszej szerokości, w celu rozpoznania pojedynczego znaku lub odstępu między znakami.
7. Użycie `model.predict(sampled_imgs)` na uzyskanych próbkach i zwrócenie listy, której elementy to rozpoznane znaki (jest ich bardzo dużo, ze względu na próbkowanie)
8. Analiza listy dla każdego wycinka. Wycinek jest dzielony na podstawie 4/5 najdłużyszch sekwencji odpowiedzi sieci o wystąpieniu odstępu. (z wyjątkiem odstępu na początku oraz na końcu sekwencji) Znaki zostają przypisane między odstępy na podstawie kryterium max z ilości odpowiedzi sieci głosujących na daną klasę w tym obszarze.
9. Formatowanie ostatecznego wyniku tak, aby rezultatem była lista tupli `(imię, nazwisko, indeks)`. Imię i nazwisko nie będą rozpoznane, więc zawsze będzie to `(None, None, indeks)`
  
## Jak uruchomić?
1. Stworzyć wirtualne środowisko `python -m venv .env`
2. Uaktywnić je: `.venv/Scripts/activate`
3. Zainstalować wymagane biblioteki `pip3 install -r ./requirements.txt`
4. Z linii poleceń wykonać `python main.py ścieżka_do_pliku_lub_katalogu`
<details>
**main.py**  wywołuje funkcję ocr.ocr(path_to_image), która bya opisana w wymaganiach zadania.

Można sprawdzić skrypt `run.sh`
</details>

## Podsumowanie
**Zaimplementowano**:
* wykrywanie linii
* rozpoznawanie numerów

Brakuje:
* rozpoznawania liter

## Przykłady
**Zdjęcie 1:**
###

<a target="_blank" href="https://github.com/yaxlie/PIRO_OCR/blob/master/res/img_1.jpg?raw=true">
  <img src="https://github.com/yaxlie/PIRO_OCR/blob/master/res/img_1.jpg?raw=true" alt="img_1.jpg" width="80">
</a>

###
**Wynik:**
```
(None, None, '725430')
(None, None, '123044')
(None, None, '135823')
(None, None, '177372')
(None, None, '589923')
(None, None, '123987')
(None, None, '181230')
(None, None, '13899')
(None, None, '457230')
(None, None, '78127')
(None, None, '22222')
```

##

**Zdjęcie 2:**
###

<a target="_blank" href="https://github.com/yaxlie/PIRO_OCR/blob/master/res/img_15.jpg?raw=true">
  <img src="https://github.com/yaxlie/PIRO_OCR/blob/master/res/img_15.jpg?raw=true" alt="img_15.jpg" width="80"/>
</a>

###
**Wynik:**
```
(None, None, '113052')
(None, None, '105552')
(None, None, '101701')
(None, None, '12311')
(None, None, '121203')
(None, None, '15431')
(None, None, '121348')
(None, None, '114105')
(None, None, '032000')
(None, None, '111112')
(None, None, '100785')
(None, None, '181762')
(None, None, '110021')
(None, None, '13716')
(None, None, '214800')
(None, None, '17400')
(None, None, '151400')
(None, None, '150012')
(None, None, '112314')
(None, None, '128316')
(None, None, '22246')
```






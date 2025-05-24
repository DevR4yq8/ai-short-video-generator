# AI Short Generator - Dokumentacja

## Spis Treści

1. [Wprowadzenie](#wprowadzenie)
2. [Główne Funkcje](#główne-funkcje)
3. [Wymagania Systemowe i Instalacja](#wymagania-systemowe-i-instalacja)
4. [Sposób Użycia (Linia Komend)](#sposób-użycia-linia-komend)
5. [Argumenty Linii Komend](#argumenty-linii-komend)
6. [Jak To Działa?](#jak-to-działa)
7. [Struktura Programu](#struktura-programu)
8. [Pliki Wyjściowe](#pliki-wyjściowe)
9. [Informacje Dodatkowe](#informacje-dodatkowe)

## 1. Wprowadzenie

AI Short Generator to program w języku Python służący do automatycznego tworzenia krótkich filmów (tzw. "shortów") z dłuższych materiałów wideo. Program wykorzystuje techniki analizy obrazu i dźwięku, w tym opcjonalnie modele sztucznej inteligencji (AI) oparte na TensorFlow, do identyfikacji i ekstrakcji najciekawszych fragmentów z oryginalnego filmu.

Głównym celem jest dostarczenie określonej liczby shortów o zadanej długości, które zachowują oryginalną rozdzielczość i proporcje obrazu materiału źródłowego. Unikalną cechą jest **inteligentne kończenie segmentów**, które dąży do zakończenia shorta, gdy "ciekawość" treści zaczyna spadać, zamiast sztywno trzymać się maksymalnej długości.

## 2. Główne Funkcje

- **Automatyczne tworzenie shortów**: Generuje zadaną liczbę krótkich klipów z długego filmu.
- **Analiza AI (opcjonalna)**: Jeśli dostępny jest TensorFlow, program wykorzystuje model MobileNetV2 do zaawansowanej analizy wizualnej. W przeciwnym razie stosuje podstawowe techniki analizy obrazu.
- **Kompleksowa analiza**: Analizuje cechy wizualne (kolor, tekstura, jasność, kontrast), cechy audio (MFCC, głośność, barwa, tempo) oraz ruch wideo.
- **Wybór "najciekawszych" momentów**: Na podstawie połączonych cech i klasteryzacji, program identyfikuje segmenty o wysokim "wyniku ciekawości".
- **Inteligentne kończenie segmentów**: Shorty mogą kończyć się wcześniej niż maksymalna dozwolona długość, jeśli analiza wskaże spadek "ciekawości" treści.
- **Konfigurowalność**: Umożliwia zdefiniowanie liczby shortów, minimalnej i maksymalnej długości każdego shortu.
- **Zachowanie oryginalnych proporcji**: Shorty są generowane z zachowaniem oryginalnej rozdzielczości i proporcji obrazu materiału źródłowego.
- **Raportowanie**: Generuje szczegółowy raport w formacie JSON zawierający informacje o procesie i wynikach, w tym o przyczynie zakończenia każdego shorta.

## 3. Wymagania Systemowe i Instalacja

**Wymagany Python**: wersja 3.11

### Biblioteki Python

Do poprawnego działania programu wymagane są następujące biblioteki:

- moviepy
- librosa
- opencv-python
- numpy
- scikit-learn
- tensorflow (opcjonalnie, dla zaawansowanej analizy AI)

Można je zainstalować za pomocą pip:

```bash
py -3.11 -m pip install moviepy librosa opencv-python numpy scikit-learn tensorflow
```

Jeśli nie chcesz korzystać z analizy opartej na TensorFlow, możesz pominąć instalację tej biblioteki. Program automatycznie wykryje jej brak i przełączy się na podstawowy tryb analizy.

## 4. Sposób Użycia (Linia Komend)

Program uruchamia się z linii komend, podając ścieżkę do pliku wideo oraz opcjonalne argumenty.

### Podstawowe użycie (generuje 10 shortów):

```bash
py -3.11 .\ai-short-video-generator.py .\film.mp4
```

### Użycie z dodatkowymi opcjami:

```bash
py -3.11 .\ai-short-video-generator.py .\film.mp4 -o /folder/wyjsciowy -c 5 --min-duration 20 --max-duration 45
```

## 5. Argumenty Linii Komend

- **video_path** (pozycyjny): Ścieżka do pliku wideo, który ma zostać przetworzony.
- **--output / -o** (opcjonalny): Nazwa folderu, w którym zostaną zapisane wygenerowane shorty oraz raport. Domyślnie: `shorts`.
- **--count / -c** (opcjonalny): Dokładna liczba shortów do wygenerowania. Program dołoży starań, aby wygenerować dokładnie tyle shortów. Domyślnie: `10`.
- **--min-duration** (opcjonalny): Minimalna długość pojedynczego shortu w sekundach. Domyślnie: `30`.
- **--max-duration** (opcjonalny): Maksymalna długość pojedynczego shortu w sekundach. Domyślnie: `60`.
- **--width** (opcjonalny): Docelowa szerokość shortu. Obecnie ignorowane, program zachowuje oryginalną szerokość. Domyślnie: `1080`.
- **--height** (opcjonalny): Docelowa wysokość shortu. Obecnie ignorowane, program zachowuje oryginalną wysokość. Domyślnie: `1920`.

## 6. Jak To Działa?

Proces generowania shortów można podzielić na kilka głównych etapów:

### 6.1. Analiza Treści (analyze_comprehensive)

1. **Ładowanie wideo**: Oryginalny plik wideo jest ładowany za pomocą moviepy.

2. **Próbkowanie**: Wideo jest analizowane poprzez pobieranie klatek w regularnych odstępach czasu (interwał próbkowania, `sample_interval_actual`, jest dynamicznie dostosowywany, domyślnie co ok. 1 sekundę dla typowych długości filmów).

3. **Ekstrakcja Cech Wizualnych**:
   - **Z TensorFlow** (`extract_features_ai`): Jeśli TensorFlow jest dostępny, każda próbkowana klatka jest przetwarzana przez model MobileNetV2 w celu ekstrakcji wektora cech opisujących zawartość wizualną.
   - **Podstawowa** (`extract_features_basic`): Jeśli TensorFlow nie jest dostępny, z klatki ekstrahowane są podstawowe cechy: histogramy kolorów w przestrzeni HSV, miara tekstury (wariancja Laplasjanu), średnia jasność i kontrast.

4. **Ekstrakcja Cech Audio** (`analyze_audio_advanced`):
   - Ścieżka dźwiękowa jest analizowana przy użyciu librosa.
   - Ekstrahowane są współczynniki MFCC (Mel-Frequency Cepstral Coefficients), głośność (RMS), jasność dźwięku (spectral centroid), tempo zmian (zero crossing rate) oraz tempo muzyczne.

5. **Analiza Ruchu** (`calculate_motion`):
   - Obliczana jest intensywność ruchu pomiędzy kolejnymi próbkowanymi klatkami przy użyciu przepływu optycznego (Optical Flow - Lucas-Kanade) lub, w przypadku problemów, różnicy absolutnej między klatkami w skali szarości.

### 6.2. Wybór Segmentów i Inteligentne Kończenie (find_best_segments_ai, find_optimal_segment_end)

1. **Normalizacja Cech**: Wszystkie zebrane cechy (wizualne, audio, ruch) są normalizowane (StandardScaler).

2. **Obliczanie "Wyniku Ciekawości"** (`calculate_interest_score`): Dla każdego momentu (próbki) obliczany jest zagregowany wynik "ciekawości", biorący pod uwagę:
   - Wariancję cech wizualnych (różnorodność wizualna)
   - Energię audio (średnia wartość absolutna pierwszych 13 współczynników MFCC)
   - Wynik analizy ruchu
   
   Wagi dla tych komponentów to odpowiednio 40% dla wizualnych, 40% dla audio i 20% dla ruchu.

3. **Klasteryzacja**: Połączone, znormalizowane cechy są poddawane klasteryzacji metodą K-Means. Celem jest znalezienie różnorodnych grup interesujących momentów.

4. **Wybór Początków Segmentów**:
   - Potencjalne początki segmentów są wybierane na podstawie najwyższych wyników ciekawości, z preferencją dla segmentów pochodzących z różnych klastrów, aby zapewnić różnorodność.
   - Program dba o to, by segmenty się znacząco nie nakładały.

5. **Inteligentne Kończenie Segmentów** (`find_optimal_segment_end`):
   - Dla każdego wybranego początku segmentu, program analizuje ewolucję "wyniku ciekawości" w kolejnych próbkach.
   - Obliczana jest bazowa ciekawość na podstawie pierwszych kilku sekund (lub odpowiadającej liczby próbek) segmentu.
   - Następnie, w ruchomym oknie (kilka próbek), monitorowana jest średnia ciekawość.
   - Jeśli średnia ciekawość w oknie spadnie poniżej pewnego progu (domyślnie 82% bazowej ciekawości, czyli spadek o 18%), segment jest kończony w tym momencie.
   - Jeśli spadek ciekawości nie nastąpi, segment osiągnie maksymalną dozwoloną długość (`max_duration`).
   - Każdy segment musi mieć co najmniej `min_duration`.

6. **Finalizacja Listy Segmentów**:
   - Jeśli początkowa selekcja z inteligentnym kończeniem nie da wymaganej liczby (`num_shorts`) shortów, algorytm próbuje dobrać kolejne najlepsze segmenty (również z inteligentnym kończeniem), a w ostateczności wypełnia brakujące miejsca losowymi fragmentami filmu o minimalnej długości.

### 6.3. Generowanie Shortów (generate_shorts)

1. **Wycinanie**: Wybrane segmenty (z określonym czasem startu i obliczoną długością) są wycinane z oryginalnego klipu wideo.

2. **Przetwarzanie (Zachowanie Proporcji)**: Funkcja `crop_to_vertical` nie zmienia proporcji ani rozdzielczości klipu. Shorty zachowują oryginalne parametry wizualne.

3. **Obsługa Audio**: Podejmowana jest próba "odświeżenia" strumienia audio dla każdego subklipu, aby zminimalizować problemy z FFMPEG.

4. **Zapis**: Każdy przetworzony segment jest zapisywany jako osobny plik MP4.

5. **Raport**: Na koniec generowany jest plik `ai_report.json`.

## 7. Struktura Programu

### Klasa AIShortGenerator

Główna logika programu zamknięta jest w klasie `AIShortGenerator`.

```python
class AIShortGenerator:
    def __init__(self, video_path, output_dir="shorts", num_shorts=10,
                 short_duration=(30, 60), target_resolution=(1080, 1920)):
        # ... inicjalizacja parametrów ...
        self.sample_interval_actual = 1.0 # Rzeczywisty interwał próbkowania
```

### Główne Metody

- `__init__(...)`: Konstruktor, inicjalizuje parametry.
- `extract_features_ai(frame)`: Ekstrahuje cechy wizualne z AI.
- `extract_features_basic(frame)`: Ekstrahuje podstawowe cechy wizualne.
- `analyze_comprehensive(video_clip)`: Przeprowadza kompleksową analizę wideo.
- `analyze_audio_advanced(video_clip)`: Analizuje ścieżkę audio.
- `calculate_motion(frame1, frame2)`: Oblicza ruch.
- `calculate_interest_score(...)`: Oblicza wynik ciekawości dla próbki.
- `find_optimal_segment_end(...)`: Określa optymalny czas zakończenia segmentu na podstawie spadku ciekawości.
- `find_best_segments_ai(analysis_data, video_clip)`: Identyfikuje i wybiera najlepsze segmenty.
- `crop_to_vertical(clip)`: Zwraca klip bez modyfikacji.
- `generate_shorts()`: Główna metoda orkiestrująca proces.
- `save_report(results, source_resolution)`: Zapisuje raport JSON.

Funkcja `main()` obsługuje parsowanie argumentów linii komend i uruchamia generator.

## 8. Pliki Wyjściowe

Po zakończeniu działania programu, w folderze wyjściowym (domyślnie `shorts/`) znajdą się:

### Pliki wideo .mp4
Wygenerowane shorty, nazwane sekwencyjnie, np. `short_01.mp4`, `short_02.mp4`, itd.

### Raport ai_report.json
Plik w formacie JSON zawierający szczegółowe informacje:

- Dane źródłowe (ścieżka, rozdzielczość)
- Typ analizy, liczba żądanych i utworzonych shortów
- Informacja o inteligentnym kończeniu i statystyki powodów zakończenia segmentów (`ending_statistics`)
- Ustawienia (zakres długości, cel formatu)
- Lista wygenerowanych shortów z detalami:
  - Nazwa pliku, czas startu, długość
  - Wynik "ciekawości" (AI score)
  - Numer klastra (jeśli dotyczy)
  - Rozdzielczość
  - Powód zakończenia segmentu (`end_reason`: np. `interest_drop`, `max_duration`, `fallback`)

## 9. Informacje Dodatkowe

- Program dąży do wygenerowania `count` shortów, stosując różne strategie, w tym inteligentne kończenie.
- Jakość i trafność wyboru segmentów zależy od materiału wideo i użycia analizy AI.
- Próg spadku ciekawości (domyślnie spadek o 18%, czyli utrzymanie 82% bazowej ciekawości) można modyfikować w metodzie `find_optimal_segment_end` (zmienna `interest_threshold`).
- Program wyświetla informacje o postępach i ewentualnych błędach w konsoli, w tym logi DEBUG dotyczące procesu znajdowania optymalnego końca segmentu.

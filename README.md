# AI Short Generator - Dokumentacja

## Spis Treści
1. [Wprowadzenie](#wprowadzenie)
2. [Główne Funkcje](#główne-funkcje)
3. [Wymagania Systemowe i Instalacja](#wymagania-systemowe-i-instalacja)
4. [Sposób Użycia (Linia Komend)](#sposób-użycia-linia-komend)
   - [Argumenty Linii Komend](#argumenty-linii-komend)
5. [Jak To Działa?](#jak-to-działa)
   - [Analiza Treści](#analiza-treści)
   - [Wybór Segmentów](#wybór-segmentów)
   - [Generowanie Shortów](#generowanie-shortów)
6. [Struktura Programu](#struktura-programu)
   - [Klasa `AIShortGenerator`](#klasa-aishortgenerator)
   - [Główne Metody](#główne-metody)
7. [Pliki Wyjściowe](#pliki-wyjściowe)
8. [Informacje Dodatkowe](#informacje-dodatkowe)

## 1. Wprowadzenie

**AI Short Generator** to program w języku Python służący do automatycznego tworzenia krótkich filmów (tzw. "shortów") z dłuższych materiałów wideo. Program wykorzystuje techniki analizy obrazu i dźwięku, w tym opcjonalnie modele sztucznej inteligencji (AI) oparte na TensorFlow, do identyfikacji i ekstrakcji najciekawszych fragmentów z oryginalnego filmu.

Głównym celem jest dostarczenie określonej liczby shortów o zadanej długości, które zachowują oryginalną rozdzielczość i proporcje obrazu materiału źródłowego.

## 2. Główne Funkcje

- **Automatyczne tworzenie shortów**: Generuje zadaną liczbę krótkich klipów z długego filmu.
- **Analiza AI (opcjonalna)**: Jeśli dostępny jest TensorFlow, program wykorzystuje model MobileNetV2 do zaawansowanej analizy wizualnej. W przeciwnym razie stosuje podstawowe techniki analizy obrazu.
- **Kompleksowa analiza**: Analizuje cechy wizualne (kolor, tekstura, jasność, kontrast), cechy audio (MFCC, głośność, barwa, tempo) oraz ruch wideo.
- **Wybór "najciekawszych" momentów**: Na podstawie połączonych cech i klasteryzacji, program identyfikuje segmenty o wysokim "wyniku ciekawości".
- **Konfigurowalność**: Umożliwia zdefiniowanie liczby shortów, minimalnej i maksymalnej długości każdego shortu.
- **Zachowanie oryginalnych proporcji**: Shorty są generowane z zachowaniem oryginalnej rozdzielczości i proporcji obrazu materiału źródłowego (funkcja kadrowania do formatu pionowego została zmodyfikowana, aby nie zmieniać proporcji).
- **Raportowanie**: Generuje szczegółowy raport w formacie JSON zawierający informacje o procesie i wynikach.

## 3. Wymagania Systemowe i Instalacja

**Wymagany Python:** wersja 3.11

**Biblioteki Python:**

Do poprawnego działania programu wymagane są następujące biblioteki:
- `moviepy`
- `librosa`
- `opencv-python`
- `numpy`
- `scikit-learn`
- `tensorflow` (opcjonalnie, dla zaawansowanej analizy AI)

Można je zainstalować za pomocą pip:

```bash
py -3.11 -m pip install moviepy librosa opencv-python numpy scikit-learn tensorflow
```

Jeśli nie chcesz korzystać z analizy opartej na TensorFlow, możesz pominąć instalację tej biblioteki. Program automatycznie wykryje jej brak i przełączy się na podstawowy tryb analizy.

## 4. Sposób Użycia (Linia Komend)

Program uruchamia się z linii komend, podając ścieżkę do pliku wideo oraz opcjonalne argumenty.

**Podstawowe użycie (generuje 10 shortów):**

```bash
py -3.11 .\ai-short-video-generator.py .\film.mp4
```

**Użycie z dodatkowymi opcjami:**

```bash
py -3.11 .\ai-short-video-generator.py .\film.mp4 -o /folder/wyjsciowy -c 5 --min-duration 20 --max-duration 45
```

### Argumenty Linii Komend

- **`video_path`** (pozycyjny): Ścieżka do pliku wideo, który ma zostać przetworzony.

- **`--output / -o`** (opcjonalny): Nazwa folderu, w którym zostaną zapisane wygenerowane shorty oraz raport. Domyślnie: `shorts`.

- **`--count / -c`** (opcjonalny): Dokładna liczba shortów do wygenerowania. Program dołoży starań, aby wygenerować dokładnie tyle shortów. Domyślnie: `10`.

- **`--min-duration`** (opcjonalny): Minimalna długość pojedynczego shortu w sekundach. Domyślnie: `30`.

- **`--max-duration`** (opcjonalny): Maksymalna długość pojedynczego shortu w sekundach. Domyślnie: `60`.

## 5. Jak To Działa?

Proces generowania shortów można podzielić na kilka głównych etapów:

### 5.1. Analiza Treści (`analyze_comprehensive`)

1. **Ładowanie wideo**: Oryginalny plik wideo jest ładowany za pomocą moviepy.

2. **Próbkowanie**: Wideo jest analizowane poprzez pobieranie klatek w regularnych odstępach czasu (interwał próbkowania jest dynamicznie dostosowywany).

3. **Ekstrakcja Cech Wizualnych**:
   - **Z TensorFlow** (`extract_features_ai`): Jeśli TensorFlow jest dostępny, każda próbkowana klatka jest przetwarzana przez model MobileNetV2 w celu ekstrakcji wektora cech opisujących zawartość wizualną.
   - **Podstawowa** (`extract_features_basic`): Jeśli TensorFlow nie jest dostępny, z klatki ekstrahowane są podstawowe cechy: histogramy kolorów w przestrzeni HSV, miara tekstury (wariancja Laplasjanu), średnia jasność i kontrast.

4. **Ekstrakcja Cech Audio** (`analyze_audio_advanced`):
   - Ścieżka dźwiękowa jest ekstrahowana i analizowana przy użyciu librosa.
   - Ekstrahowane są współczynniki MFCC (Mel-Frequency Cepstral Coefficients), głośność (RMS), jasność dźwięku (spectral centroid), tempo zmian (zero crossing rate) oraz tempo muzyczne.

5. **Analiza Ruchu** (`calculate_motion`):
   - Obliczana jest intensywność ruchu pomiędzy kolejnymi próbkowanymi klatkami przy użyciu przepływu optycznego (Optical Flow - Lucas-Kanade) lub, w przypadku problemów, różnicy absolutnej między klatkami w skali szarości.

### 5.2. Wybór Segmentów (`find_best_segments_ai`)

1. **Normalizacja Cech**: Wszystkie zebrane cechy (wizualne, audio, ruch) są normalizowane (`StandardScaler`).

2. **Obliczanie "Wyniku Ciekawości"**: Dla każdego momentu (próbki) obliczany jest zagregowany wynik "ciekawości", biorący pod uwagę:
   - Wariancję cech (różnorodność)
   - Intensywność (odległość od średniej)
   - Energię audio (średnia wartość absolutna MFCC)
   - Wynik analizy ruchu

   Cechy audio mają przypisaną większą wagę (domyślnie 60%) w połączonym wektorze cech, a cechy wizualne i ruch odpowiednio mniejsze (30% i 10%).

3. **Klasteryzacja**: Połączone, znormalizowane cechy są poddawane klasteryzacji metodą K-Means. Celem jest znalezienie różnorodnych grup interesujących momentów.

4. **Wybór Najlepszych Segmentów**:
   - Segmenty są wybierane na podstawie najwyższych wyników ciekawości, z preferencją dla segmentów pochodzących z różnych klastrów, aby zapewnić różnorodność.
   - Długość każdego segmentu jest losowana z zadanego przedziału (`min_duration`, `max_duration`).
   - Program dba o to, by segmenty się nie nakładały (z pewnym buforem).
   - Jeśli początkowa selekcja nie da wymaganej liczby (`num_shorts`) shortów, algorytm próbuje dobrać kolejne najlepsze segmenty, a w ostateczności wypełnia brakujące miejsca losowymi fragmentami filmu.

### 5.3. Generowanie Shortów (`generate_shorts`)

1. **Wycinanie**: Wybrane segmenty są wycinane z oryginalnego klipu wideo za pomocą moviepy.

2. **Przetwarzanie (Zachowanie Proporcji)**: Funkcja `crop_to_vertical` w obecnej wersji nie zmienia proporcji ani rozdzielczości klipu. Zwraca oryginalny wycięty segment. Oznacza to, że shorty będą miały takie same proporcje i rozdzielczość jak materiał źródłowy.

3. **Zapis**: Każdy przetworzony segment jest zapisywany jako osobny plik MP4 w folderze wyjściowym. Używane są standardowe kodeki (libx264 dla wideo, aac dla audio) dla zapewnienia szerokiej kompatybilności.

4. **Raport**: Na koniec generowany jest plik `ai_report.json` podsumowujący proces.

## 6. Struktura Programu

### Klasa `AIShortGenerator`

Główna logika programu zamknięta jest w klasie `AIShortGenerator`.

```python
class AIShortGenerator:
    def __init__(self, video_path, output_dir="shorts", num_shorts=10,
                 short_duration=(30, 60), target_resolution=(1080, 1920)):
        # ... inicjalizacja parametrów ...
```

### Główne Metody

- **`__init__(...)`**: Konstruktor, inicjalizuje ścieżkę do wideo, folder wyjściowy, liczbę shortów, zakres ich trwania oraz (historycznie) docelową rozdzielczość.

- **`extract_features_ai(frame)`**: Ekstrahuje cechy wizualne z klatki przy użyciu modelu AI (TensorFlow).

- **`extract_features_basic(frame)`**: Ekstrahuje podstawowe cechy wizualne z klatki (bez AI).

- **`analyze_comprehensive(video_clip)`**: Przeprowadza kompleksową analizę wideo, zbierając cechy wizualne, audio i ruchowe.

- **`analyze_audio_advanced(video_clip)`**: Analizuje ścieżkę audio, ekstrahując m.in. MFCC, RMS, tempo.

- **`calculate_motion(frame1, frame2)`**: Oblicza intensywność ruchu między dwiema klatkami.

- **`find_best_segments_ai(analysis_data, video_clip)`**: Identyfikuje i wybiera najlepsze segmenty na podstawie zebranych danych analitycznych i klasteryzacji.

- **`crop_to_vertical(clip)`**: **Ważne:** Obecnie ta metoda zwraca klip bez modyfikacji, zachowując jego oryginalne proporcje i rozdzielczość. Nie wykonuje kadrowania do formatu pionowego.

- **`generate_shorts()`**: Główna metoda orkiestrująca cały proces: od analizy, przez wybór segmentów, aż po generowanie i zapis plików shortów.

- **`save_report(results, source_resolution)`**: Zapisuje szczegółowy raport z działania programu w formacie JSON.

Funkcja `main()` obsługuje parsowanie argumentów linii komend i uruchamia generator.

## 7. Pliki Wyjściowe

Po zakończeniu działania programu, w folderze wyjściowym (domyślnie `shorts/` lub zdefiniowanym przez opcję `-o`) znajdą się:

### Pliki wideo `.mp4`
Wygenerowane shorty, nazwane sekwencyjnie, np. `short_01.mp4`, `short_02.mp4`, itd. Ich rozdzielczość i proporcje będą takie same jak w oryginalnym filmie.

### Raport `ai_report.json`
Plik w formacie JSON zawierający szczegółowe informacje:

- Ścieżka do oryginalnego wideo i jego rozdzielczość
- Typ użytej analizy (zaawansowana z AI lub podstawowa)
- Żądana i faktycznie utworzona liczba shortów
- Ustawienia (zakres długości shortów, cel formatu wyjściowego)
- Lista wygenerowanych shortów wraz z ich:
  - Nazwą pliku
  - Czasem rozpoczęcia w oryginalnym filmie
  - Długością
  - Wynikiem "ciekawości" (AI score)
  - Numerem klastra (jeśli dotyczy)
  - Rozdzielczością

## 8. Informacje Dodatkowe

- Program jest zaprojektowany tak, aby wygenerować dokładnie tyle shortów, ile zażądano (`--count`). W tym celu stosuje różne strategie doboru segmentów, włącznie z wypełnianiem brakujących miejsc losowymi fragmentami, jeśli analiza nie wskaże wystarczającej liczby "ciekawych" momentów.

- Jakość i trafność wyboru segmentów zależy od wielu czynników, w tym od charakteru materiału wideo oraz od tego, czy używana jest zaawansowana analiza AI (TensorFlow), czy podstawowa.

- Podczas działania program wyświetla w konsoli informacje o postępach poszczególnych etapów analizy i generowania.

- Wagi przypisane różnym typom cech (wizualne, audio, ruch) przy obliczaniu "wyniku ciekawości" oraz w połączonym wektorze cech można modyfikować w kodzie źródłowym (w metodzie `find_best_segments_ai`), aby dostosować kryteria wyboru "najciekawszych" momentów.

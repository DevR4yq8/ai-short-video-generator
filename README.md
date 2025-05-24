# AI Short Generator - Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Main Features](#main-features)
3. [System Requirements and Installation](#system-requirements-and-installation)
4. [Usage (Command Line)](#usage-command-line)
5. [Command Line Arguments](#command-line-arguments)
6. [How It Works](#how-it-works)
7. [Program Structure](#program-structure)
8. [Output Files](#output-files)
9. [Additional Information](#additional-information)

## 1. Introduction

AI Short Generator is a Python program designed for automatically creating short videos (so-called "shorts") from longer video materials. The program uses image and sound analysis techniques, including optionally artificial intelligence (AI) models based on TensorFlow, to identify and extract the most interesting fragments from the original video.

The main goal is to provide a specified number of shorts with a given duration that preserve the original resolution and aspect ratio of the source material. A unique feature is **intelligent segment ending**, which aims to end the short when the "interest" of the content begins to decline, instead of rigidly adhering to the maximum duration.

## 2. Main Features

- **Automatic short creation**: Generates a specified number of short clips from a long video.
- **AI analysis (optional)**: If TensorFlow is available, the program uses the MobileNetV2 model for advanced visual analysis. Otherwise, it applies basic image analysis techniques.
- **Comprehensive analysis**: Analyzes visual features (color, texture, brightness, contrast), audio features (MFCC, volume, timbre, tempo), and video motion.
- **Selection of "most interesting" moments**: Based on combined features and clustering, the program identifies segments with high "interest scores".
- **Intelligent segment ending**: Shorts can end earlier than the maximum allowed duration if analysis indicates a drop in content "interest".
- **Configurability**: Allows defining the number of shorts, minimum and maximum duration of each short.
- **Preserving original proportions**: Shorts are generated while maintaining the original resolution and aspect ratio of the source material.
- **Reporting**: Generates a detailed JSON report containing information about the process and results, including the reason for ending each short.

## 3. System Requirements and Installation

**Required Python**: version 3.11

### Python Libraries

The following libraries are required for proper program operation:

- moviepy
- librosa
- opencv-python
- numpy
- scikit-learn
- tensorflow (optionally, for advanced AI analysis)

They can be installed using pip:

```bash
py -3.11 -m pip install moviepy librosa opencv-python numpy scikit-learn tensorflow
```

If you don't want to use TensorFlow-based analysis, you can skip installing this library. The program will automatically detect its absence and switch to basic analysis mode.

## 4. Usage (Command Line)

The program is run from the command line, providing the path to the video file and optional arguments.

### Basic usage (generates 10 shorts):

```bash
py -3.11 .\ai-short-video-generator.py .\film.mp4
```

### Usage with additional options:

```bash
py -3.11 .\ai-short-video-generator.py .\film.mp4 -o /output/folder -c 5 --min-duration 20 --max-duration 45
```

## 5. Command Line Arguments

- **video_path** (positional): Path to the video file to be processed.
- **--output / -o** (optional): Name of the folder where generated shorts and report will be saved. Default: `shorts`.
- **--count / -c** (optional): Exact number of shorts to generate. The program will strive to generate exactly this many shorts. Default: `10`.
- **--min-duration** (optional): Minimum duration of a single short in seconds. Default: `30`.
- **--max-duration** (optional): Maximum duration of a single short in seconds. Default: `60`.
- **--width** (optional): Target width of the short. Currently ignored, program preserves original width. Default: `1080`.
- **--height** (optional): Target height of the short. Currently ignored, program preserves original height. Default: `1920`.

## 6. How It Works

The short generation process can be divided into several main stages:

### 6.1. Content Analysis (analyze_comprehensive)

1. **Video loading**: The original video file is loaded using moviepy.

2. **Sampling**: The video is analyzed by extracting frames at regular time intervals (sampling interval, `sample_interval_actual`, is dynamically adjusted, typically about every 1 second for typical video lengths).

3. **Visual Feature Extraction**:
   - **With TensorFlow** (`extract_features_ai`): If TensorFlow is available, each sampled frame is processed by the MobileNetV2 model to extract a feature vector describing visual content.
   - **Basic** (`extract_features_basic`): If TensorFlow is not available, basic features are extracted from the frame: color histograms in HSV space, texture measure (Laplacian variance), average brightness and contrast.

4. **Audio Feature Extraction** (`analyze_audio_advanced`):
   - The audio track is analyzed using librosa.
   - MFCC coefficients (Mel-Frequency Cepstral Coefficients), volume (RMS), sound brightness (spectral centroid), rate of change (zero crossing rate), and musical tempo are extracted.

5. **Motion Analysis** (`calculate_motion`):
   - Motion intensity is calculated between consecutive sampled frames using optical flow (Optical Flow - Lucas-Kanade) or, in case of problems, absolute difference between grayscale frames.

### 6.2. Segment Selection and Intelligent Ending (find_best_segments_ai, find_optimal_segment_end)

1. **Feature Normalization**: All collected features (visual, audio, motion) are normalized (StandardScaler).

2. **"Interest Score" Calculation** (`calculate_interest_score`): For each moment (sample), an aggregated "interest" score is calculated, taking into account:
   - Visual feature variance (visual diversity)
   - Audio energy (average absolute value of the first 13 MFCC coefficients)
   - Motion analysis score
   
   Weights for these components are 40% for visual, 40% for audio, and 20% for motion respectively.

3. **Clustering**: Combined, normalized features are subjected to K-Means clustering. The goal is to find diverse groups of interesting moments.

4. **Segment Start Selection**:
   - Potential segment starts are selected based on highest interest scores, with preference for segments from different clusters to ensure diversity.
   - The program ensures that segments don't significantly overlap.

5. **Intelligent Segment Ending** (`find_optimal_segment_end`):
   - For each selected segment start, the program analyzes the evolution of "interest score" in subsequent samples.
   - Base interest is calculated based on the first few seconds (or corresponding number of samples) of the segment.
   - Then, in a moving window (several samples), average interest is monitored.
   - If the average interest in the window drops below a certain threshold (default 82% of base interest, i.e., an 18% drop), the segment is ended at that point.
   - If interest doesn't drop, the segment reaches maximum allowed duration (`max_duration`).
   - Each segment must have at least `min_duration`.

6. **Segment List Finalization**:
   - If initial selection with intelligent ending doesn't yield the required number (`num_shorts`) of shorts, the algorithm tries to select additional best segments (also with intelligent ending), and ultimately fills missing slots with random video fragments of minimum duration.

### 6.3. Short Generation (generate_shorts)

1. **Cutting**: Selected segments (with specified start time and calculated duration) are cut from the original video clip.

2. **Processing (Preserving Proportions)**: The `crop_to_vertical` function doesn't change clip proportions or resolution. Shorts preserve original visual parameters.

3. **Audio Handling**: An attempt is made to "refresh" the audio stream for each subclip to minimize FFMPEG issues.

4. **Saving**: Each processed segment is saved as a separate MP4 file.

5. **Report**: Finally, an `ai_report.json` file is generated.

## 7. Program Structure

### AIShortGenerator Class

The main program logic is enclosed in the `AIShortGenerator` class.

```python
class AIShortGenerator:
    def __init__(self, video_path, output_dir="shorts", num_shorts=10,
                 short_duration=(30, 60), target_resolution=(1080, 1920)):
        # ... parameter initialization ...
        self.sample_interval_actual = 1.0 # Actual sampling interval
```

### Main Methods

- `__init__(...)`: Constructor, initializes parameters.
- `extract_features_ai(frame)`: Extracts visual features with AI.
- `extract_features_basic(frame)`: Extracts basic visual features.
- `analyze_comprehensive(video_clip)`: Performs comprehensive video analysis.
- `analyze_audio_advanced(video_clip)`: Analyzes audio track.
- `calculate_motion(frame1, frame2)`: Calculates motion.
- `calculate_interest_score(...)`: Calculates interest score for a sample.
- `find_optimal_segment_end(...)`: Determines optimal segment end time based on interest drop.
- `find_best_segments_ai(analysis_data, video_clip)`: Identifies and selects best segments.
- `crop_to_vertical(clip)`: Returns clip without modification.
- `generate_shorts()`: Main method orchestrating the process.
- `save_report(results, source_resolution)`: Saves JSON report.

The `main()` function handles command line argument parsing and runs the generator.

## 8. Output Files

After program completion, the output folder (default `shorts/`) will contain:

### Video files .mp4
Generated shorts, named sequentially, e.g., `short_01.mp4`, `short_02.mp4`, etc.

### Report ai_report.json
JSON format file containing detailed information:

- Source data (path, resolution)
- Analysis type, number of requested and created shorts
- Information about intelligent ending and statistics of segment ending reasons (`ending_statistics`)
- Settings (duration range, format target)
- List of generated shorts with details:
  - File name, start time, duration
  - "Interest" score (AI score)
  - Cluster number (if applicable)
  - Resolution
  - Segment ending reason (`end_reason`: e.g., `interest_drop`, `max_duration`, `fallback`)

## 9. Additional Information

- The program strives to generate `count` shorts using various strategies, including intelligent ending.
- Quality and accuracy of segment selection depends on video material and AI analysis usage.
- The interest drop threshold (default 18% drop, i.e., maintaining 82% of base interest) can be modified in the `find_optimal_segment_end` method (`interest_threshold` variable).
- The program displays progress information and potential errors in the console, including DEBUG logs regarding the optimal segment end finding process.

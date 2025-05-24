import os
import cv2
import numpy as np
import librosa
from moviepy.editor import VideoFileClip, CompositeVideoClip, AudioFileClip
import argparse
from datetime import timedelta
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    TENSORFLOW_AVAILABLE = True
    print("🧠 TensorFlow dostępne - używam zaawansowanej analizy AI")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("📝 TensorFlow niedostępne - używam podstawowej analizy")

class AIShortGenerator:
    def __init__(self, video_path, output_dir="shorts", num_shorts=10, 
                 short_duration=(30, 60), target_resolution=(1080, 1920)):
        self.video_path = video_path
        self.output_dir = output_dir
        self.num_shorts = num_shorts
        self.min_duration = short_duration[0]
        self.max_duration = short_duration[1]
        self.target_width = target_resolution[0] 
        self.target_height = target_resolution[1]
        
        os.makedirs(output_dir, exist_ok=True)
        
        if TENSORFLOW_AVAILABLE:
            self.ai_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
            print("🤖 Załadowano model AI do analizy obrazu")
        else:
            self.ai_model = None
    
    def extract_features_ai(self, frame):
        """Wyciągnij cechy obrazu używając AI"""
        if not TENSORFLOW_AVAILABLE or self.ai_model is None:
            return self.extract_features_basic(frame)
        
        try:
            img = cv2.resize(frame, (224, 224))
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            features = self.ai_model.predict(img, verbose=0)
            return features.flatten()
        except Exception as e:
            print(f"⚠️ Błąd AI, używam podstawowej analizy: {e}")
            return self.extract_features_basic(frame)
    
    def extract_features_basic(self, frame):
        """Podstawowa analiza cech obrazu bez AI"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [50], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [50], [0, 256])
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        texture = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(hsv[:,:,2])
        contrast = np.std(hsv[:,:,2])
        features = np.concatenate([
            hist_h.flatten(), hist_s.flatten(), hist_v.flatten(),
            [texture, brightness, contrast]
        ])
        return features
    
    def analyze_comprehensive(self, video_clip):
        """Kompleksowa analiza filmu z AI"""
        print("🧠 Rozpoczynam kompleksową analizę AI...")
        
        duration = video_clip.duration
        sample_interval = max(1.0, duration / 600)
        
        timestamps = []
        visual_features = []
        audio_features_list = [] 
        motion_scores = []
        
        print(f"📊 Analizuję film ({duration:.1f}s) z interwałem {sample_interval:.1f}s")
        
        print("🎵 Analizuję audio...")
        audio_times, audio_data = self.analyze_audio_advanced(video_clip)

        audio_features_available = True
        default_audio_feature_shape = 17 
        if audio_data is None or len(audio_data) == 0:
            print("⚠️ Brak danych audio do analizy. Cechy audio będą zerowe.")
            audio_features_available = False
        elif len(audio_data) > 0 and hasattr(audio_data, 'shape') and len(audio_data.shape) > 1:
             default_audio_feature_shape = audio_data.shape[1]


        print("🎬 Analizuję obraz i ruch...")
        prev_frame = None
        
        for i, t in enumerate(np.arange(0, duration - 1, sample_interval)):
            try:
                frame = video_clip.get_frame(t)
                visual_feat = self.extract_features_ai(frame)
                visual_features.append(visual_feat)
                
                if prev_frame is not None:
                    motion = self.calculate_motion(prev_frame, frame)
                    motion = 0.0 if motion is None else motion
                else:
                    motion = 0.0
                motion_scores.append(motion)
                
                audio_feat_current = np.zeros(default_audio_feature_shape)
                if audio_features_available and len(audio_times) > 0:
                    audio_idx = np.argmin(np.abs(audio_times - t))
                    if audio_idx < len(audio_data):
                        audio_feat_current = audio_data[audio_idx]
                
                audio_features_list.append(audio_feat_current)
                timestamps.append(t)
                prev_frame = frame.copy()
                
                if i % 50 == 0:
                    progress = (i * sample_interval / duration) * 100
                    print(f"  Progress: {progress:.1f}%")
                    
            except Exception as e:
                print(f"⚠️ Błąd przy analizie t={t:.1f}s: {e}")
                if len(timestamps) > 0:
                    if len(visual_features) > len(timestamps): visual_features.pop()
                    if len(motion_scores) > len(timestamps): motion_scores.pop()
                    if len(audio_features_list) > len(timestamps): audio_features_list.pop()
                continue
        
        return {
            'timestamps': np.array(timestamps),
            'visual_features': np.array(visual_features),
            'audio_features': np.array(audio_features_list), 
            'motion_scores': np.array(motion_scores)
        }
    
    def analyze_audio_advanced(self, video_clip):
        """Zaawansowana analiza audio z MFCC"""
        y, sr = None, None
        try:
            print(f"🎵 Analizuję audio z pliku: {self.video_path} używając librosa")
            y, sr = librosa.load(self.video_path, sr=None, mono=True)
        except Exception as e:
            print(f"⚠️ Błąd podczas ładowania audio przez librosa: {e}. Próbuję metody awaryjnej z moviepy...")
            if hasattr(video_clip, 'audio') and video_clip.audio:
                temp_audio_fallback = "temp_audio_fallback.wav"
                try:
                    video_clip.audio.write_audiofile(temp_audio_fallback, codec='pcm_s16le', verbose=False, logger=None)
                    y, sr = librosa.load(temp_audio_fallback)
                except Exception as e_fallback:
                     print(f"⚠️ Błąd metody awaryjnej moviepy: {e_fallback}")
                finally:
                    if os.path.exists(temp_audio_fallback):
                        try:
                            os.remove(temp_audio_fallback)
                        except Exception as e_remove:
                            print(f"⚠️ Nie udało się usunąć {temp_audio_fallback}: {e_remove}")
            else:
                print("⛔ Wideo nie posiada ścieżki audio lub obiekt video_clip nie ma atrybutu audio.")
        
        if y is None or len(y) == 0 or sr is None:
            print("⚠️ Nie udało się załadować danych audio lub próbkowania. Zwracam puste dane.")
            return np.array([]), np.array([])

        hop_length = sr // 4 
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
        
        try:
            tempo_val, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
            tempo_val = float(tempo_val) if np.isscalar(tempo_val) else float(np.mean(tempo_val))
        except Exception as e:
            print(f"⚠️ Błąd analizy tempa: {e}")
            tempo_val = 120.0
        
        audio_features_out = []
        times = librosa.frames_to_time(np.arange(mfccs.shape[1]), sr=sr, hop_length=hop_length)
        
        for i in range(mfccs.shape[1]):
            rms_val = rms[i] if i < len(rms) else 0.0
            centroid_val = centroid[i] if i < len(centroid) else 0.0
            zcr_val = zcr[i] if i < len(zcr) else 0.0
            
            mfcc_vals = mfccs[:, i].flatten()
            other_vals = np.array([rms_val, centroid_val, zcr_val, tempo_val])
            
            features = np.concatenate([mfcc_vals, other_vals])
            audio_features_out.append(features)
        
        return times, np.array(audio_features_out)

    def calculate_motion(self, frame1, frame2):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
        
        motion_intensity = 0.0
        if p0 is not None:
            lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            try:
                p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)
                if p1 is not None and st is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    if len(good_new) > 0:
                        distances = np.linalg.norm(good_new - good_old, axis=1)
                        motion_intensity = np.mean(distances)
                    else:
                        diff = cv2.absdiff(gray1, gray2)
                        motion_intensity = np.mean(diff) / 2.0 
                else:
                    diff = cv2.absdiff(gray1, gray2)
                    motion_intensity = np.mean(diff)
            except cv2.error: 
                diff = cv2.absdiff(gray1, gray2)
                motion_intensity = np.mean(diff)
            except Exception: 
                diff = cv2.absdiff(gray1, gray2)
                motion_intensity = np.mean(diff)
        else:
            diff = cv2.absdiff(gray1, gray2)
            motion_intensity = np.mean(diff)
        return motion_intensity
    
    def calculate_interest_score(self, visual_features_vec, audio_features_vec, motion_score_val):
        diversity = np.var(visual_features_vec) if visual_features_vec is not None and len(visual_features_vec) > 0 else 0.0
        
        audio_energy = 0.0
        if audio_features_vec is not None and len(audio_features_vec) > 0:
            num_mfcc = 13
            if len(audio_features_vec) >= num_mfcc:
                audio_energy = np.mean(np.abs(audio_features_vec[:num_mfcc]))
            else:
                audio_energy = np.mean(np.abs(audio_features_vec))
        
        current_motion_score = motion_score_val if motion_score_val is not None else 0.0

        interest = (diversity * 0.4 + audio_energy * 0.4 + current_motion_score * 0.2)
        return interest
    
    def find_optimal_segment_end(self, start_idx, analysis_data, video_duration):
        timestamps = analysis_data['timestamps']
        visual_features = analysis_data['visual_features']
        audio_features = analysis_data['audio_features']
        motion_scores = analysis_data['motion_scores']

        if start_idx >= len(timestamps):
             print(f"⚠️ start_idx ({start_idx}) poza zakresem timestamps ({len(timestamps)})")
             return min(timestamps[len(timestamps)-1] + self.min_duration, video_duration) if len(timestamps)>0 else video_duration

        start_time = timestamps[start_idx]
        min_end_time = start_time + self.min_duration
        max_end_time = min(start_time + self.max_duration, video_duration)
        
        min_end_idx = np.argmin(np.abs(timestamps - min_end_time)) if len(timestamps) > 0 else 0
        max_end_idx = np.argmin(np.abs(timestamps - max_end_time)) if len(timestamps) > 0 else 0
        
        min_end_idx = min(min_end_idx, len(timestamps) - 1) if len(timestamps) > 0 else 0
        max_end_idx = min(max_end_idx, len(timestamps) - 1) if len(timestamps) > 0 else 0
            
        print(f"DEBUG: find_optimal_segment_end dla start_time={start_time:.2f}s (idx={start_idx})") # DEBUG

        baseline_end_idx = min(start_idx + int(5 / (timestamps[1]-timestamps[0] if len(timestamps)>1 else 1)), len(timestamps) - 1)
        baseline_scores = []
        
        for i in range(start_idx, baseline_end_idx + 1):
            if i < len(visual_features) and i < len(audio_features) and i < len(motion_scores):
                score = self.calculate_interest_score(
                    visual_features[i], audio_features[i], motion_scores[i]
                )
                baseline_scores.append(score)
        
        if not baseline_scores:
            print(f"DEBUG: Brak baseline_scores dla start_idx={start_idx}. Zwracam max_end_time={max_end_time:.2f}s") # DEBUG
            return max_end_time 
            
        baseline_interest = np.mean(baseline_scores) if baseline_scores else 0
        interest_threshold = baseline_interest * 0.82
        print(f"DEBUG: baseline_interest={baseline_interest:.4f}, interest_threshold={interest_threshold:.4f}") # DEBUG 
        
        optimal_end_time = max_end_time
        window_size = 3

        print(f"DEBUG: min_end_idx={min_end_idx}, max_end_idx={max_end_idx}, window_size_samples={window_size}") # DEBUG
        
        for i in range(min_end_idx, max_end_idx + 1 - (window_size -1) ):
                
            window_scores = []
            current_window_times = [] # DEBUG
            for j in range(i, i + window_size):
                if j >= len(timestamps): break
                score = self.calculate_interest_score(
                    visual_features[j], audio_features[j], motion_scores[j]
                )
                window_scores.append(score)
                current_window_times.append(timestamps[j]) # DEBUG
            
            if window_scores:
                window_interest = np.mean(window_scores)
                print(f"DEBUG:  idx={i}, time_window_start={current_window_times[0]:.2f}s, window_interest={window_interest:.4f}") # DEBUG
                if window_interest < interest_threshold:
                    optimal_end_time = timestamps[i] if i < len(timestamps) else max_end_time
                    print(f"DEBUG: SPADEK CIEKAWOŚCI! Optimal_end_time={optimal_end_time:.2f}s (idx={i})") # DEBUG
                    break
        
        print(f"DEBUG: Końcowy optimal_end_time={optimal_end_time:.2f}s dla start_idx={start_idx}") # DEBUG
        return optimal_end_time
    
    def find_best_segments_ai(self, analysis_data, video_clip):
        print("🔍 AI wybiera najciekawsze momenty z inteligentnym kończeniem...")
        
        timestamps = analysis_data.get('timestamps', np.array([]))
        visual_features = analysis_data.get('visual_features', np.array([]))
        audio_features = analysis_data.get('audio_features', np.array([]))
        motion_scores = analysis_data.get('motion_scores', np.array([]))

        if timestamps.size == 0 or visual_features.size == 0 or audio_features.size == 0 or motion_scores.size == 0:
            print("⚠️ Brak wystarczających danych z analizy (timestamps, visual, audio, or motion is empty). Generowanie losowych.")
            segments = []
            for _ in range(self.num_shorts):
                max_start_fallback = video_clip.duration - self.min_duration 
                if max_start_fallback <= 0: break 
                start_time_fallback = np.random.uniform(0, max_start_fallback)
                segments.append({
                    'start': start_time_fallback, 'duration': self.min_duration,
                    'interest_score': 0.0, 'cluster': -3, 'end_reason': 'fallback_no_analysis_data'
                })
            return segments[:self.num_shorts]

        scaler_visual = StandardScaler()
        scaler_audio = StandardScaler()
        scaler_motion = StandardScaler()
        
        visual_norm = scaler_visual.fit_transform(visual_features) if visual_features.ndim > 1 and visual_features.shape[0] > 0 else np.array([])
        audio_norm = scaler_audio.fit_transform(audio_features) if audio_features.ndim > 1 and audio_features.shape[0] > 0 else np.array([])
        motion_norm = scaler_motion.fit_transform(motion_scores.reshape(-1, 1)) if motion_scores.size > 0 else np.array([])
        
        feature_parts = []
        if visual_norm.size > 0: feature_parts.append(visual_norm * 0.4)
        if audio_norm.size > 0: feature_parts.append(audio_norm * 0.4)
        if motion_norm.size > 0: feature_parts.append(motion_norm * 0.2)

        if not feature_parts:
            print("⚠️ Nie można połączyć cech - wszystkie są puste po normalizacji. Generowanie losowych segmentów.")
            segments = []
            for _ in range(self.num_shorts):
                max_start_fallback2 = video_clip.duration - self.min_duration
                if max_start_fallback2 <= 0: break
                start_time_fallback2 = np.random.uniform(0, max_start_fallback2)
                segments.append({
                    'start': start_time_fallback2, 'duration': self.min_duration,
                    'interest_score': 0.0, 'cluster': -3, 'end_reason': 'fallback_no_combined_features'
                })
            return segments[:self.num_shorts]

        combined_features = np.hstack(feature_parts)
        if combined_features.size == 0:
            print("⚠️ Połączone cechy są puste. Generowanie losowych segmentów.")
            segments = []
            for _ in range(self.num_shorts):
                max_start_fallback3 = video_clip.duration - self.min_duration
                if max_start_fallback3 <= 0: break
                start_time_fallback3 = np.random.uniform(0, max_start_fallback3)
                segments.append({
                    'start': start_time_fallback3, 'duration': self.min_duration,
                    'interest_score': 0.0, 'cluster': -3, 'end_reason': 'fallback_empty_combined'
                })
            return segments[:self.num_shorts]

        interest_scores = []
        num_samples = min(len(timestamps), len(visual_features), len(audio_features), len(motion_scores))
        for i in range(num_samples): 
            score = self.calculate_interest_score(
                visual_features[i], audio_features[i], motion_scores[i]
            )
            interest_scores.append(score)
        interest_scores = np.array(interest_scores)
        if interest_scores.size == 0:
            print("⚠️ Nie udało się obliczyć wyników ciekawości. Generowanie losowych segmentów.")
            segments = []
            for _ in range(self.num_shorts):
                max_start_fallback4 = video_clip.duration - self.min_duration
                if max_start_fallback4 <= 0: break
                start_time_fallback4 = np.random.uniform(0, max_start_fallback4)
                segments.append({
                    'start': start_time_fallback4, 'duration': self.min_duration,
                    'interest_score': 0.0, 'cluster': -3, 'end_reason': 'fallback_no_interest_scores'
                })
            return segments[:self.num_shorts]

        n_clusters = min(self.num_shorts * 3, len(combined_features))
        if n_clusters > 1 and len(combined_features) >= n_clusters : 
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto') 
            clusters = kmeans.fit_predict(combined_features)
        else:
            clusters = np.zeros(len(timestamps), dtype=int) 
        
        segments = []
        used_clusters = set()
        sorted_indices = np.argsort(interest_scores)[::-1] if interest_scores.size > 0 else np.array([])
        
        for idx in sorted_indices:
            if idx >= num_samples: continue
            if len(segments) >= self.num_shorts * 2 : break 

            cluster_val = clusters[idx] if idx < len(clusters) else -1 
            if cluster_val in used_clusters and len(segments) < self.num_shorts : 
                 continue
                
            start_time = timestamps[idx]
            min_gap = self.max_duration + 5 
            if any(abs(start_time - seg['start']) < min_gap for seg in segments):
                continue
            
            optimal_end_time = self.find_optimal_segment_end(idx, analysis_data, video_clip.duration)
            duration = optimal_end_time - start_time
            duration = np.clip(duration, self.min_duration, self.max_duration)
            
            if start_time + duration <= video_clip.duration:
                segments.append({
                    'start': start_time, 'duration': duration,
                    'interest_score': interest_scores[idx],
                    'cluster': int(cluster_val), 
                    'end_reason': 'interest_drop' if optimal_end_time < start_time + self.max_duration else 'max_duration'
                })
                used_clusters.add(cluster_val)
        
        current_segments_count = len(segments)
        if current_segments_count < self.num_shorts:
            additional_needed = self.num_shorts - current_segments_count
            segments = sorted(segments, key=lambda x: x['interest_score'], reverse=True)

            potential_adds = []
            for idx in sorted_indices:
                if idx >= num_samples: continue
                if len(potential_adds) >= additional_needed * 2: break 

                start_time = timestamps[idx]
                is_too_close = False
                for seg in segments:
                    if abs(start_time - seg['start']) < self.min_duration: 
                        is_too_close = True
                        break
                if is_too_close:
                    continue

                optimal_end_time = self.find_optimal_segment_end(idx, analysis_data, video_clip.duration)
                duration = optimal_end_time - start_time
                duration = np.clip(duration, self.min_duration, self.max_duration)

                if start_time + duration <= video_clip.duration:
                    potential_adds.append({
                        'start': start_time, 'duration': duration,
                        'interest_score': interest_scores[idx], 'cluster': -1, 
                        'end_reason': 'interest_drop' if optimal_end_time < start_time + self.max_duration else 'max_duration'
                    })
            
            potential_adds = sorted(potential_adds, key=lambda x: x['interest_score'], reverse=True)
            for add_seg in potential_adds:
                if len(segments) >= self.num_shorts: break
                is_too_close_final = False
                for seg in segments: 
                    if abs(add_seg['start'] - seg['start']) < self.min_duration:
                        is_too_close_final = True
                        break
                if not is_too_close_final:
                    segments.append(add_seg)

        while len(segments) < self.num_shorts:
            max_start = video_clip.duration - self.min_duration 
            if max_start <= 0:
                print("⚠️ Film za krótki na dodanie więcej segmentów fallback.")
                break 
            start_time = np.random.uniform(0, max_start)
            is_too_close_fallback = False
            for seg in segments:
                if abs(start_time - seg['start']) < self.min_duration:
                     is_too_close_fallback = True
                     break
            if is_too_close_fallback:
                continue 

            segments.append({
                'start': start_time, 'duration': self.min_duration, 
                'interest_score': 0.0, 'cluster': -2, 'end_reason': 'fallback'
            })
        
        segments = sorted(segments, key=lambda x: x['interest_score'], reverse=True)
        segments = segments[:self.num_shorts]
        segments = sorted(segments, key=lambda x: x['start']) 
        
        print(f"✅ Wybrano {len(segments)} najciekawszych segmentów:")
        for i, seg in enumerate(segments):
            print(f"  Short {i+1}: {timedelta(seconds=int(seg['start']))} "
                  f"({seg['duration']:.1f}s, wynik: {seg['interest_score']:.3f}, "
                  f"koniec: {seg['end_reason']})")
        return segments
    
    def crop_to_vertical(self, clip):
        print("INFO: Zachowuję oryginalną rozdzielczość i proporcje klipu.")
        return clip

    def generate_shorts(self):
        print(f"🚀 AI Generator - tworzę dokładnie {self.num_shorts} shortów z: {self.video_path}")
        
        video_clip = None
        try:
            video_clip = VideoFileClip(self.video_path)
        except Exception as e:
            print(f"❌ Nie można załadować pliku wideo: {self.video_path}. Błąd: {e}")
            return

        original_width, original_height = video_clip.size
        print(f"📹 Długość filmu: {timedelta(seconds=int(video_clip.duration))}, Rozdzielczość: {original_width}x{original_height}")
        
        min_required_duration_total = self.num_shorts * self.min_duration
        if video_clip.duration < self.min_duration: 
             print(f"⚠️ Film jest krótszy ({video_clip.duration:.1f}s) niż minimalna długość shorta ({self.min_duration}s). Nie można utworzyć shortów.")
             video_clip.close()
             return

        if video_clip.duration < min_required_duration_total:
            print(f"⚠️ Film może być za krótki ({video_clip.duration:.1f}s) "
                  f"dla {self.num_shorts} shortów ({min_required_duration_total}s minimum sumarycznie).")
            print("Kontynuuję z możliwym nakładaniem segmentów lub mniejszą liczbą shortów...")
        
        analysis_data = self.analyze_comprehensive(video_clip)
        
        if not isinstance(analysis_data, dict) or not analysis_data:
            print("❌ Analiza wideo nie zwróciła prawidłowych danych (analysis_data nie jest słownikiem lub jest pusty). Przerywam generowanie shortów.")
            if video_clip: video_clip.close() 
            return

        timestamps_array = analysis_data.get('timestamps')
        if not isinstance(timestamps_array, np.ndarray) or timestamps_array.size == 0:
            print("❌ Analiza wideo nie zwróciła danych dla 'timestamps' lub są one puste. Przerywam generowanie shortów.")
            if video_clip: video_clip.close()
            return

        segments = self.find_best_segments_ai(analysis_data, video_clip)
        
        if len(segments) > self.num_shorts:
            segments = sorted(segments, key=lambda x: x['interest_score'], reverse=True)[:self.num_shorts]
            segments = sorted(segments, key=lambda x: x['start'])

        actual_num_shorts_to_generate = min(len(segments), self.num_shorts)
        if actual_num_shorts_to_generate == 0:
            print("⚠️ Nie znaleziono żadnych odpowiednich segmentów do utworzenia shortów.")
            video_clip.close()
            return

        print(f"🎬 Generuję {actual_num_shorts_to_generate} shortów z inteligentnym kończeniem...")
        
        results = []
        for i, segment in enumerate(segments[:actual_num_shorts_to_generate]): 
            print(f"🎥 Tworzę short {i+1}/{actual_num_shorts_to_generate} "
                  f"(start: {timedelta(seconds=int(segment['start']))}, "
                  f"długość: {segment['duration']:.1f}s)")
            
            short_clip_segment = None 
            processed_clip = None   
            final_short_clip_info = None 

            try:
                start_time = segment['start']
                duration = segment['duration']
                
                if start_time + duration > video_clip.duration:
                    duration = video_clip.duration - start_time
                
                if duration < 1.0: 
                    print(f"⚠️ Segment {i+1} za krótki ({duration:.1f}s) po korekcie, pomijam.")
                    continue
                
                short_clip_segment = video_clip.subclip(start_time, start_time + duration)

                if short_clip_segment.audio:
                    try:
                        print(f"INFO: Odświeżam audio dla shorta {i+1}...")
                        fresh_audio_clip = AudioFileClip(self.video_path).subclip(start_time, start_time + duration)
                        short_clip_segment = short_clip_segment.set_audio(fresh_audio_clip)
                        if not short_clip_segment.audio:
                            print(f"WARN: Nie udało się ustawić odświeżonego audio dla shorta {i+1}.")
                    except Exception as e_audio_refresh:
                        print(f"WARN: Błąd podczas odświeżania audio dla shorta {i+1}: {e_audio_refresh}")
                elif hasattr(video_clip, 'audio') and video_clip.audio:
                    print(f"WARN: Subklip {i+1} powinien mieć audio, ale go nie ma przed odświeżeniem.")
                
                if hasattr(video_clip, 'audio') and video_clip.audio and (not hasattr(short_clip_segment, 'audio') or not short_clip_segment.audio):
                    print(f"⚠️ Subclip dla shorta {i+1} stracił audio. Próbuję dołączyć audio z oryginału.")
                    try:
                        if video_clip.audio:
                            audio_subclip = video_clip.audio.subclip(start_time, start_time + duration)
                            short_clip_segment = short_clip_segment.set_audio(audio_subclip)
                            if not short_clip_segment.audio:
                                print(f"⚠️ Nadal brak audio dla shorta {i+1} po próbie dołączenia.")
                        else:
                            print(f"⚠️ Oryginalny klip nie ma audio do skopiowania dla shorta {i+1}.")
                    except Exception as e_audio:
                        print(f"⚠️ Błąd przy próbie dołączenia audio do shorta {i+1}: {e_audio}")

                processed_clip = self.crop_to_vertical(short_clip_segment) 
                
                output_path = os.path.join(self.output_dir, f"short_{i+1:02d}.mp4")
                
                processed_clip.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    preset='medium', 
                    ffmpeg_params=['-profile:v', 'baseline', '-level', '3.0', '-pix_fmt', 'yuv420p', '-crf', '23'], 
                    remove_temp=True,
                    verbose=False, 
                    logger=None    
                )
                
                final_short_clip_info = VideoFileClip(output_path)
                short_width, short_height = final_short_clip_info.size
                
                results.append({
                    'file': output_path,
                    'start_time': f"{timedelta(seconds=int(start_time))}",
                    'duration': f"{duration:.1f}s",
                    'ai_score': f"{segment.get('interest_score', 0.0):.3f}",
                    'cluster': segment.get('cluster', -9), 
                    'resolution': f"{short_width}x{short_height}",
                    'end_reason': segment.get('end_reason', 'unknown')
                })
                
                print(f"✅ Zapisano: {output_path} (Rozdzielczość: {short_width}x{short_height}, "
                      f"Koniec: {segment.get('end_reason', 'unknown')})")
            
            except Exception as e:
                print(f"❌ Błąd przy tworzeniu short {i+1}: {e}")
                import traceback
                traceback.print_exc()
            finally: 
                if final_short_clip_info: final_short_clip_info.close()
                if processed_clip: processed_clip.close()
                if short_clip_segment: short_clip_segment.close()

        if video_clip: video_clip.close()
        
        if len(results) != self.num_shorts and len(results) != actual_num_shorts_to_generate :
             print(f"⚠️ Docelowo miało być {self.num_shorts} shortów, wygenerowano {actual_num_shorts_to_generate}, zapisano {len(results)}.")
        
        self.save_report(results, f"{original_width}x{original_height}")
        
        print(f"\n🎉 AI Generator zakończony! Utworzono {len(results)} shortów w: {self.output_dir}")
        if any(r.get('end_reason') == 'interest_drop' for r in results):
            print(f"🧠 Niektóre shorty zostały inteligentnie zakończone gdy ciekawość spadła!")
    
    def save_report(self, results, source_resolution):
        report_path = os.path.join(self.output_dir, "ai_report.json")
        end_reasons = {}
        for result in results:
            reason = result.get('end_reason', 'unknown')
            end_reasons[reason] = end_reasons.get(reason, 0) + 1
        
        report = {
            'source_video': self.video_path,
            'source_resolution': source_resolution,
            'ai_analysis': 'Zaawansowana analiza z AI' if TENSORFLOW_AVAILABLE else 'Podstawowa analiza',
            'total_shorts_requested': self.num_shorts,
            'total_shorts_created': len(results),
            'intelligent_ending': True,
            'ending_statistics': end_reasons,
            'settings': {
                'duration_range': f"{self.min_duration}-{self.max_duration}s",
                'output_format_goal': "Oryginalne proporcje i rozdzielczość",
                'ai_features': 'visual+audio+motion analysis + intelligent ending'
            },
            'shorts': results
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"📊 Raport AI zapisany: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='AI Generator shortów - inteligentne kończenie na spadku ciekawości')
    parser.add_argument('video_path', help='Ścieżka do pliku wideo')
    parser.add_argument('--output', '-o', default='shorts', help='Folder wyjściowy')
    parser.add_argument('--count', '-c', type=int, default=10, help='DOKŁADNA liczba shortów (domyślnie: 10)')
    parser.add_argument('--min-duration', type=int, default=30, help='Min długość shortu')
    parser.add_argument('--max-duration', type=int, default=60, help='Max długość shortu')
    parser.add_argument('--width', type=int, default=1080, help='Docelowa szerokość (obecnie ignorowane, zachowuje oryginał)')
    parser.add_argument('--height', type=int, default=1920, help='Docelowa wysokość (obecnie ignorowane, zachowuje oryginał)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"❌ Plik nie istnieje: {args.video_path}")
        return
    
    print(f"🎯 AI spróbuje zagwarantować do {args.count} shortów z inteligentnym kończeniem!")
    
    generator = AIShortGenerator(
        video_path=args.video_path,
        output_dir=args.output,
        num_shorts=args.count,
        short_duration=(args.min_duration, args.max_duration),
        target_resolution=(args.width, args.height) 
    )
    
    try:
        generator.generate_shorts()
    except KeyboardInterrupt:
        print("\n⏹️ Przerwano przez użytkownika")
    except Exception as e:
        print(f"❌ Nieoczekiwany globalny błąd: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
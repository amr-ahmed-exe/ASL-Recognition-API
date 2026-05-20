import os
import json
import asyncio
from collections import Counter, deque
import numpy as np
import torch
import enchant
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from train_gcn_model import SignLanguageGCNModel

# ---------------------------------------------------------------------------
# 1. إعداد السيرفر والمكتبات
# ---------------------------------------------------------------------------
app = FastAPI(title="ASL Recognition API (PyTorch GCN)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dictionary = enchant.Dict("en_US")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 2. تحميل الموديل (PyTorch GCN)
# ---------------------------------------------------------------------------
print("Loading PyTorch GCN Model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLanguageGCNModel(num_classes=24).to(device)

MODEL_PATH = os.path.join(SCRIPT_DIR, "sign_language_gcn_model.pth")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# الحروف اللي الموديل متدرب عليها (مفيش J ومفيش Z)
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 
          'T', 'U', 'V', 'W', 'X', 'Y']

# ---------------------------------------------------------------------------
# 3. فئات معالجة وتنعيم البيانات (Landmark Filter & Typing Engine)
# ---------------------------------------------------------------------------
class LandmarkFilter:
    """Smoothes incoming landmarks using an Exponential Moving Average (EMA) filter to eliminate jitter."""
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self.last_smoothed = None

    def smooth(self, pts: list) -> list:
        if not pts:
            return pts
        
        current = np.array(pts, dtype=np.float32)
        if self.last_smoothed is None:
            self.last_smoothed = current
            return pts
        
        # EMA Filter formula
        smoothed = self.alpha * current + (1 - self.alpha) * self.last_smoothed
        self.last_smoothed = smoothed
        return smoothed.tolist()

    def reset(self):
        self.last_smoothed = None


class ASLTypingEngine:
    """Manages prediction smoothing, confidence thresholds, smart keyboard debouncing, auto-repeat, and auto-space."""
    def __init__(self, required_consecutive: int = 5, min_confidence: float = 0.35, 
                 repeat_delay: int = 20, no_hand_timeout: int = 35):
        self.required_consecutive = required_consecutive
        self.min_confidence = min_confidence
        self.repeat_delay = repeat_delay
        self.no_hand_timeout = no_hand_timeout

        self.current_word = ""
        self.current_sentence = ""

        # Probability smoothing window
        self.prob_history = deque(maxlen=7)
        
        # Typing state variables
        self.last_predicted_letter = None
        self.consecutive_count = 0
        self.last_appended_letter = None
        self.has_had_no_hand_since_last_append = True
        self.no_hand_frames = 0
        self.auto_spaced_committed = False

    def process_prediction(self, raw_letter: str, confidence: float, probabilities: list) -> tuple:
        """
        Processes a raw frame prediction.
        Returns (confirmed_letter, current_word, display_text)
        """
        confirmed_letter = None
        self.no_hand_frames = 0  # Reset no-hand counter
        self.auto_spaced_committed = False

        if not raw_letter or confidence < self.min_confidence:
            # If the raw prediction is too weak, treat it as a frame with no valid prediction
            self.prob_history.clear()
            self.consecutive_count = 0
            self.last_predicted_letter = None
            return None, self.current_word, self.get_display_text()

        # Add to probability history
        self.prob_history.append(probabilities)
        
        # Calculate rolling average of probabilities
        avg_probs = np.mean(self.prob_history, axis=0)
        best_idx = np.argmax(avg_probs)
        smoothed_letter = LABELS[best_idx]
        smoothed_confidence = avg_probs[best_idx]

        # Overwrite dynamic J/Z if motion triggers it
        if raw_letter in ("J", "Z"):
            smoothed_letter = raw_letter
            smoothed_confidence = confidence

        # Apply minimum confidence to smoothed prediction
        if smoothed_confidence < self.min_confidence:
            return None, self.current_word, self.get_display_text()

        # Update consecutive frame count
        if smoothed_letter == self.last_predicted_letter:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 1
            self.last_predicted_letter = smoothed_letter

        # When the letter stabilizes for enough consecutive frames
        if self.consecutive_count >= self.required_consecutive:
            # Smart debounce: check if it matches the last appended letter
            if smoothed_letter == self.last_appended_letter:
                # Allow repeat ONLY if hand was removed/relaxed (no_hand flag) OR if held for repeat_delay frames
                if self.has_had_no_hand_since_last_append or self.consecutive_count >= self.repeat_delay:
                    confirmed_letter = smoothed_letter
                    self._append_letter(confirmed_letter)
            else:
                confirmed_letter = smoothed_letter
                self._append_letter(confirmed_letter)

        return confirmed_letter, self.current_word, self.get_display_text()

    def process_no_hand(self) -> tuple:
        """
        Called when no hand is detected in a frame.
        Handles auto-space / auto-commit timeout.
        """
        self.prob_history.clear()
        self.consecutive_count = 0
        self.last_predicted_letter = None
        self.has_had_no_hand_since_last_append = True
        
        self.no_hand_frames += 1
        
        # Auto-space on hand drop timeout
        if self.current_word and self.no_hand_frames >= self.no_hand_timeout and not self.auto_spaced_committed:
            self.current_sentence += (" " + self.current_word) if self.current_sentence else self.current_word
            self.current_word = ""
            self.auto_spaced_committed = True
            
        return self.current_word, self.get_display_text()

    def _append_letter(self, letter: str):
        if letter == "Space":
            self.current_sentence += (" " + self.current_word) if self.current_sentence else self.current_word
            self.current_word = ""
        else:
            self.current_word += letter
            
        self.last_appended_letter = letter
        self.has_had_no_hand_since_last_append = False
        self.consecutive_count = 0  # Reset counter after typing to initiate new hold cycle

    def get_display_text(self) -> str:
        display_text = self.current_sentence
        if self.current_word:
            display_text += (" " + self.current_word) if display_text else self.current_word
        return display_text

    def commit_word(self, selected_word: str):
        self.current_sentence += (" " + selected_word) if self.current_sentence else selected_word
        self.current_word = ""
        self.prob_history.clear()
        self.consecutive_count = 0
        self.last_predicted_letter = None

    def clear(self):
        self.current_word = ""
        self.current_sentence = ""
        self.prob_history.clear()
        self.last_predicted_letter = None
        self.consecutive_count = 0
        self.last_appended_letter = None
        self.has_had_no_hand_since_last_append = True
        self.no_hand_frames = 0
        self.auto_spaced_committed = False


# ---------------------------------------------------------------------------
# 4. دالة التوقع الأساسية (PyTorch Inference & Dynamic J/Z Detection)
# ---------------------------------------------------------------------------
def predict_from_pytorch(pts, j_z_history=None):
    flat_landmarks = []
    
    # تحويل النقط لـ 63 رقم خام (X, Y, Z)
    for p in pts:
        x = float(p[0])
        y = float(p[1])
        z = float(p[2]) if len(p) > 2 else 0.0 
        flat_landmarks.extend([x, y, z])
        
    if len(flat_landmarks) != 63:
        return None, 0.0, None, "invalid_landmarks_shape"

    # التوقع باستخدام الـ Transformer
    landmarks_tensor = torch.tensor(flat_landmarks, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(landmarks_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0)
        max_prob, predicted_idx = torch.max(probabilities, 0)
        
    letter = LABELS[predicted_idx.item()]
    confidence = max_prob.item()
    prob_list = probabilities.cpu().tolist()
    
    # ── J/Z Motion Detection ──────────────────────────────────────────
    if j_z_history and len(j_z_history) >= 8:
        # Calculate dynamic hand scale (distance between Wrist [0] and Middle finger MCP [9])
        p0 = np.array(pts[0])
        p9 = np.array(pts[9])
        hand_scale = np.linalg.norm(p9 - p0)
        if hand_scale < 0.01:
            hand_scale = 0.2  # Fallback to standard scale if calculation fails

        # ▶ J — حركة الخنصر: ينزل لأسفل ويعمل خطاف (ي)
        # pts[20] = pinky tip
        if letter == "I":
            pinky = [h[1] for h in j_z_history]
            y_displacement = pinky[-1][1] - pinky[0][1]   # موجب = نزول
            x_displacement = abs(pinky[-1][0] - pinky[0][0])

            # Normalize displacements by hand scale
            y_disp_norm = y_displacement / hand_scale
            x_disp_norm = x_displacement / hand_scale

            # Base thresholds normalized by reference hand scale 0.2
            if y_disp_norm > 0.4 and x_disp_norm > 0.125:
                letter = "J"

        # ▶ Z — حركة السبابة: يمين → أسفل-يسار → يمين (تغيير اتجاه في X مرتين)
        # pts[8] = index finger tip
        elif letter in ("D", "U", "V"):
            index = [h[0] for h in j_z_history]

            # velocities normalized by hand scale
            x_vels_norm = [(index[i][0] - index[i-1][0]) / hand_scale for i in range(1, len(index))]

            # نعد كام مرة الاتجاه اتغير
            direction_changes = 0
            cur_dir = None
            for v_norm in x_vels_norm:
                if abs(v_norm) > 0.025:           # base threshold normalized
                    new_dir = 1 if v_norm > 0 else -1
                    if cur_dir is not None and new_dir != cur_dir:
                        direction_changes += 1
                    cur_dir = new_dir

            y_total = abs(index[-1][1] - index[0][1])
            x_total = abs(index[-1][0] - index[0][0])
            y_total_norm = y_total / hand_scale
            x_total_norm = x_total / hand_scale

            if direction_changes >= 2 and y_total_norm > 0.2 and x_total_norm > 0.15:
                letter = "Z"
    
    return letter, confidence, prob_list, "success"


# ---------------------------------------------------------------------------
# 5. دالة الاقتراحات (Spell Check)
# ---------------------------------------------------------------------------
def get_suggestions(word: str, max_count: int = 4) -> list[str]:
    word = word.strip().upper()
    if not word: return []
    try:
        word_lower = word.lower()
        if dictionary.check(word_lower):
            return [word]
        raw = dictionary.suggest(word_lower)
        filtered = [s.upper() for s in raw if s.lower().startswith(word_lower)]
        if not filtered:
            filtered = [s.upper() for s in raw]
        return filtered[:max_count]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# 6. الـ WebSocket (للاتصال المباشر مع الموبايل)
# ---------------------------------------------------------------------------
@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    print("Flutter Client Connected!")

    loop = asyncio.get_running_loop()
    
    # Initialize our filter and engine
    landmark_filter = LandmarkFilter(alpha=0.4)
    typing_engine = ASLTypingEngine(
        required_consecutive=5, 
        min_confidence=0.35, 
        repeat_delay=20, 
        no_hand_timeout=35
    )
    j_z_history = deque(maxlen=15)

    last_word_for_suggestions = ""
    cached_suggestions: list = []

    try:
        while True:
            data_text = await websocket.receive_text()

            # أوامر التحكم من الموبايل
            if data_text == "CLEAR":
                typing_engine.clear()
                landmark_filter.reset()
                j_z_history.clear()
                continue
            
            if data_text.startswith("COMMIT:"):
                selected_word = data_text.split(":", 1)[1]
                typing_engine.commit_word(selected_word)
                landmark_filter.reset()
                j_z_history.clear()
                continue

            # استلام النقط
            try:
                pts = json.loads(data_text)
            except (json.JSONDecodeError, ValueError):
                continue

            if not pts or len(pts) != 21:
                status = "no_hand"
                letter = None
                confidence = 0.0
                
                # Process the frame with no hand in the typing engine (for auto-space timer)
                current_word, display_text = typing_engine.process_no_hand()
                confirmed_letter = None
            else:
                # Apply EMA landmark jitter filter
                smoothed_pts = landmark_filter.smooth(pts)
                
                # تتبع السبابة والخنصر للـ J و Z
                j_z_history.append((smoothed_pts[8], smoothed_pts[20]))
                history_snapshot = list(j_z_history)

                # التوقع
                letter, confidence, prob_list, status = await loop.run_in_executor(
                    None, predict_from_pytorch, smoothed_pts, history_snapshot
                )

                if letter and status == "success":
                    confirmed_letter, current_word, display_text = typing_engine.process_prediction(
                        letter, confidence, prob_list
                    )
                else:
                    # Treat as no_hand frame if predict failed
                    current_word, display_text = typing_engine.process_no_hand()
                    confirmed_letter = None

            # اقتراح الكلمات — بيتحسب بس لما الكلمة تتغير (cache)
            if current_word and len(current_word) >= 2:
                if current_word != last_word_for_suggestions:
                    cached_suggestions = get_suggestions(current_word)
                    last_word_for_suggestions = current_word
                suggestions = cached_suggestions
            else:
                suggestions = []
                last_word_for_suggestions = ""

            # إرسال النتيجة للموبايل
            await websocket.send_json({
                "status": status,
                "raw_prediction": letter,
                "confidence": confidence,
                "confirmed_letter": confirmed_letter,
                "current_word": current_word,
                "final_word": display_text,
                "suggestions": suggestions
            })

    except WebSocketDisconnect:
        print("Flutter Client Disconnected")
    except Exception as e:
        print(f"Error: {e}")

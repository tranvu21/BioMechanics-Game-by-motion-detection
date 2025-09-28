
import time
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set

import cv2
import mediapipe as mp
import numpy as np
from pynput.keyboard import Controller as KBController, Key

# ----------------------------
# Data classes
# ----------------------------

@dataclass
class Speeds:
    left: float = 0.0
    right: float = 0.0
    avg: float = 0.0

@dataclass
class Direction:
    label: str  # "left" | "right" | "straight"
    dz: float

@dataclass
class Thresholds:
    idle_speed: float = 120.0       # below -> Idle
    walk_speed: float = 200.0       # avg above -> Walking
    run_speed: float = 300.0        # BOTH hands above -> Running
    yaw_thresh: float = 0.30        # torso yaw threshold from shoulder z-diff
    raise_margin: float = 0.05      # wrist must be this much above shoulder for "raised"

# ----------------------------
# Keyboard controller
# ----------------------------

class KeyController:
    def __init__(self, use_shift: bool = True):
        self.kb = KBController()
        self.use_shift = use_shift
        self.active_keys: Set = set()
        # motion_modules.py  — inside KeyController.__init__
        self.MOVE_KEYS = {"FORWARD": "w", "LEFT": "a", "RIGHT": "d"}
        self.COLLECT_KEY = 'e'
        self.SHIFT_KEY = Key.shift
        self.SPACE_KEY = Key.space    
  

    def apply(self, target: Set):
        # Release keys not in target
        for k in list(self.active_keys):
            if k not in target:
                try: self.kb.release(k)
                except Exception: pass
                self.active_keys.remove(k)
        # Press keys newly needed
        for k in target:
            if k not in self.active_keys:
                try: self.kb.press(k)
                except Exception: pass
                self.active_keys.add(k)

    def release_all(self):
        for k in list(self.active_keys):
            try: self.kb.release(k)
            except Exception: pass
        self.active_keys.clear()

# ----------------------------
# Latching policy (debounce/hold keys until change confirmed)
# ----------------------------

class ActionLatch:
    """
    Holds previously active key-set until a new key-set is observed
    for `switch_frames` consecutive frames. When the detected set is empty,
    we keep holding the previous keys until it's empty for `none_grace` frames.
    """
    def __init__(self, switch_frames: int = 4, none_grace: int = 8):
        self.switch_frames = max(1, switch_frames)
        self.none_grace = max(0, none_grace)

        self.current: Set = set()
        self.pending: Optional[Set] = None
        self.pending_count = 0
        self.none_count = 0
        self.initialized = False

    def update(self, new_keys: Set) -> Set:
        new_f = frozenset(new_keys)

        if not self.initialized:
            self.current = set(new_keys)
            self.initialized = True
            self.pending = None
            self.pending_count = 0
            self.none_count = 0
            return set(self.current)

        cur_f = frozenset(self.current)

        if new_f == cur_f:
            self.pending = None
            self.pending_count = 0
            self.none_count = 0 if new_keys else self.none_count + 1
            if not self.current and self.none_count < self.none_grace:
                pass
            return set(self.current)

        if not new_keys:
            self.pending = None
            self.pending_count = 0
            self.none_count += 1
            if self.none_count >= self.none_grace:
                self.current = set()
            return set(self.current)

        self.none_count = 0
        if self.pending is None or frozenset(self.pending) != new_f:
            self.pending = set(new_keys)
            self.pending_count = 1
            return set(self.current)
        else:
            self.pending_count += 1
            if self.pending_count >= self.switch_frames:
                self.current = set(self.pending)
                self.pending = None
                self.pending_count = 0
            return set(self.current)

# ----------------------------
# Pose estimator (wrists speeds + torso yaw)
# ----------------------------

class PoseEstimator:
    def __init__(self, min_det: float = 0.5, min_track: float = 0.5, buffer: int = 20):
        self.mp_pose = mp.solutions.pose
        self.drawer = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=min_det, min_tracking_confidence=min_track)

        self.prev_time = time.time()
        self.prev_left_wrist_px: Optional[Tuple[int, int]] = None
        self.prev_right_wrist_px: Optional[Tuple[int, int]] = None

        self.left_buffer: List[float] = []
        self.right_buffer: List[float] = []
        self.avg_buffer: List[float] = []
        self.BUFFER = max(1, buffer)

        self._last_pose_proto = None

    @staticmethod
    def _dist(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def process(self, bgr_image):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.pose.process(rgb)

        self._last_pose_proto = res.pose_landmarks

        speeds = Speeds()
        direction = Direction("straight", 0.0)
        landmarks = None

        if res.pose_landmarks:
            h, w, _ = bgr_image.shape
            landmarks = res.pose_landmarks.landmark

            lw = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].x * w),
                  int(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y * h))
            rw = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].x * w),
                  int(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y * h))

            cur_time = time.time()
            dt = max(1e-6, cur_time - self.prev_time)
            self.prev_time = cur_time

            if self.prev_left_wrist_px and self.prev_right_wrist_px:
                ls = self._dist(lw, self.prev_left_wrist_px) / dt
                rs = self._dist(rw, self.prev_right_wrist_px) / dt
                avg = 0.5 * (ls + rs)

                self.left_buffer.append(ls);  self.right_buffer.append(rs);  self.avg_buffer.append(avg)
                if len(self.left_buffer) > self.BUFFER: self.left_buffer.pop(0)
                if len(self.right_buffer) > self.BUFFER: self.right_buffer.pop(0)
                if len(self.avg_buffer) > self.BUFFER: self.avg_buffer.pop(0)

                speeds = Speeds(
                    left=float(np.mean(self.left_buffer)),
                    right=float(np.mean(self.right_buffer)),
                    avg=float(np.mean(self.avg_buffer))
                )

            self.prev_left_wrist_px, self.prev_right_wrist_px = lw, rw

            # Torso yaw from shoulders z
            ls = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            rs = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            if (ls.visibility or 1.0) >= 0.5 and (rs.visibility or 1.0) >= 0.5:
                dz = rs.z - ls.z
                direction = Direction("straight", dz)

        return landmarks, speeds, direction

    def draw(self, image):
        if self._last_pose_proto is not None:
            self.drawer.draw_landmarks(image, self._last_pose_proto, self.mp_pose.POSE_CONNECTIONS)

# ----------------------------
# Hands estimator (closed fist detection)
# ----------------------------

class HandsEstimator:
    def __init__(self, min_det: float = 0.5, min_track: float = 0.5, max_hands: int = 2):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_det,
            min_tracking_confidence=min_track,
        )

    @staticmethod
    def _is_fist_closed(single_hand) -> bool:
        """
        Simple heuristic:
          - For each finger, tip is below PIP (y larger -> lower on image)
          - Thumb tip near IP joint (touch-ish)
        """
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        count = 0
        for tip, pip in zip(tips, pips):
            if single_hand.landmark[tip].y > single_hand.landmark[pip].y:
                count += 1
        thumb_close = (abs(single_hand.landmark[4].x - single_hand.landmark[3].x) < 0.05 and
                       abs(single_hand.landmark[4].y - single_hand.landmark[3].y) < 0.05)
        return (count >= 3) and thumb_close

    def any_closed(self, bgr_image) -> bool:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            return False
        for hand in res.multi_hand_landmarks:
            if self._is_fist_closed(hand):
                return True
        return False

# ----------------------------
# Motion classifier
# ----------------------------

class MotionClassifier:
    def __init__(self, thresholds: Thresholds):
        self.t = thresholds
        self.mp_pose = mp.solutions.pose

    def yaw_label(self, direction: Direction) -> str:
        if direction.dz < -self.t.yaw_thresh: return "right"
        if direction.dz >  self.t.yaw_thresh: return "left"
        return "straight"

    def classify(self, speeds: Speeds, direction: Direction) -> str:
        # Overall intensity
        if speeds.avg < self.t.idle_speed:
            return "Idle"

        yaw = self.yaw_label(direction)
        dir_label = "FORWARD" if yaw == "straight" else yaw.upper()

        both_fast = (speeds.left >= self.t.run_speed) and (speeds.right >= self.t.run_speed)
        if both_fast:
            return f"Running {dir_label}"

        if speeds.avg >= self.t.walk_speed:
            return f"Walking {dir_label}"

        return "Idle"

    def wrist_raised(self, landmarks) -> bool:
        if landmarks is None:
            return False
        lw = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        rw = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        ls = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        # y smaller = higher on image
        raised_left  = (lw.y < (ls.y - self.t.raise_margin))
        raised_right = (rw.y < (rs.y - self.t.raise_margin))
        return bool(raised_left or raised_right)
    # motion_modules.py  — inside class MotionClassifier
    def both_hands_up(self, landmarks) -> bool:
        """
        True when BOTH arms are raised straight up:
        - wrists above head level, and
        - elbows above shoulders (reduces false positives)
        """
        if landmarks is None:
            return False

        mp_pose = self.mp_pose
        lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        le = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        re = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Head reference (y smaller = higher). Use face points if present; else shoulders - offset.
        head_ids = [
            mp_pose.PoseLandmark.NOSE,
            mp_pose.PoseLandmark.LEFT_EYE,
            mp_pose.PoseLandmark.RIGHT_EYE,
            mp_pose.PoseLandmark.LEFT_EAR,
            mp_pose.PoseLandmark.RIGHT_EAR,
        ]
        head_ys = [landmarks[i].y for i in head_ids]
        head_y = min(head_ys) if head_ys else (min(ls.y, rs.y) - 0.10)

        margin = self.t.raise_margin  # reuse your configured margin
        left_up  = (lw.y < head_y) and (le.y < (ls.y - margin))
        right_up = (rw.y < head_y) and (re.y < (rs.y - margin))
        return left_up and right_up



# ----------------------------
# Game mapper (activity -> keys)
# ----------------------------

class GameMapper:
    def __init__(self, keyboard: KeyController):
        self.kb = keyboard

    def map_activity(self, activity: str) -> Set:
        k = set()
        # Movement
        if activity.startswith("Running "):
            if "LEFT" in activity:
                k.add(self.kb.MOVE_KEYS["LEFT"])
            elif "RIGHT" in activity:
                k.add(self.kb.MOVE_KEYS["RIGHT"])
            else:
                k.add(self.kb.MOVE_KEYS["FORWARD"])
            if self.kb.use_shift:
                k.add(self.kb.SHIFT_KEY)

        elif activity.startswith("Walking "):
            if "LEFT" in activity:
                k.add(self.kb.MOVE_KEYS["LEFT"])
            elif "RIGHT" in activity:
                k.add(self.kb.MOVE_KEYS["RIGHT"])
            else:
                k.add(self.kb.MOVE_KEYS["FORWARD"])

        # Idle -> no movement keys
        return k

    def maybe_add_collect(self, keys: Set, collect: bool) -> Set:
        if collect:
            keys = set(keys)
            keys.add(self.kb.COLLECT_KEY)  # 'e'
        return keys

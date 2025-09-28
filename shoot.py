# run_mouse_hold.py
# Palm (open)  -> HOLD Right Mouse (RMB)
# Fist         -> HOLD Left + Right (LMB + RMB) until palm opens again or hand disappears
# Default camera id = 1

import argparse
import math
from typing import Tuple

import cv2
import mediapipe as mp
from pynput.mouse import Controller as MSController, Button as MSButton


# ---------- geometry helpers ----------
def angle_deg(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Angle ABC (at B) using 2D coordinates in degrees."""
    (ax, ay), (bx, by), (cx, cy) = a, b, c
    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 180.0
    cosang = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cosang))


def lm_xy(lm, w, h):
    return (lm.x * w, lm.y * h)


# ---------- gesture detector ----------
class GestureDetector:
    def __init__(self, extended_thr: float = 160.0, curled_thr: float = 110.0):
        """
        extended_thr: angle at PIP/DIP >= this => finger extended (straight)
        curled_thr:   angle at PIP <= this     => finger curled
        """
        self.extended_thr = extended_thr
        self.curled_thr = curled_thr
        self.mp_hands = mp.solutions.hands

    def _finger_angles(self, lms, w, h, mcp_i, pip_i, dip_i, tip_i):
        mcp = lm_xy(lms[mcp_i], w, h)
        pip = lm_xy(lms[pip_i], w, h)
        dip = lm_xy(lms[dip_i], w, h)
        tip = lm_xy(lms[tip_i], w, h)
        a_pip = angle_deg(mcp, pip, dip)
        a_dip = angle_deg(pip, dip, tip)
        return a_pip, a_dip

    def finger_extended(self, lms, w, h, mcp_i, pip_i, dip_i, tip_i) -> bool:
        a_pip, a_dip = self._finger_angles(lms, w, h, mcp_i, pip_i, dip_i, tip_i)
        return (a_pip >= self.extended_thr) and (a_dip >= self.extended_thr)

    def finger_curled(self, lms, w, h, mcp_i, pip_i, dip_i, tip_i) -> bool:
        a_pip, _ = self._finger_angles(lms, w, h, mcp_i, pip_i, dip_i, tip_i)
        return a_pip <= self.curled_thr

    def is_open_palm(self, lms, w, h) -> bool:
        """Index..pinky all extended (thumb ignored)."""
        idx = (5, 6, 7, 8)
        mid = (9, 10, 11, 12)
        rng = (13, 14, 15, 16)
        pky = (17, 18, 19, 20)
        extended = [
            self.finger_extended(lms, w, h, *idx),
            self.finger_extended(lms, w, h, *mid),
            self.finger_extended(lms, w, h, *rng),
            self.finger_extended(lms, w, h, *pky),
        ]
        return sum(extended) >= 4  # all four

    def is_fist(self, lms, w, h) -> bool:
        """Index..pinky curled (thumb ignored). Allow slight tolerance."""
        idx = (5, 6, 7, 8)
        mid = (9, 10, 11, 12)
        rng = (13, 14, 15, 16)
        pky = (17, 18, 19, 20)
        curled = [
            self.finger_curled(lms, w, h, *idx),
            self.finger_curled(lms, w, h, *mid),
            self.finger_curled(lms, w, h, *rng),
            self.finger_curled(lms, w, h, *pky),
        ]
        return sum(curled) >= 3


# ---------- mouse wrapper ----------
class MouseHold:
    def __init__(self):
        self.ms = MSController()
        self.lmb_down = False
        self.rmb_down = False

    def hold_left(self, on: bool):
        if on and not self.lmb_down:
            try:
                self.ms.press(MSButton.left)
                self.lmb_down = True
            except Exception:
                pass
        elif not on and self.lmb_down:
            try:
                self.ms.release(MSButton.left)
                self.lmb_down = False
            except Exception:
                pass

    def hold_right(self, on: bool):
        if on and not self.rmb_down:
            try:
                self.ms.press(MSButton.right)
                self.rmb_down = True
            except Exception:
                pass
        elif not on and self.rmb_down:
            try:
                self.ms.release(MSButton.right)
                self.rmb_down = False
            except Exception:
                pass


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Palm -> hold RMB | Fist -> hold LMB+RMB (cam=1)")
    ap.add_argument("--cam", type=int, default=1, help="Camera index (default 1)")
    ap.add_argument("--confirm_on", type=int, default=3, help="Frames to confirm PALM/FIST")
    ap.add_argument("--none_for_release", type=int, default=5, help="Frames of no-hand before releasing")
    args = ap.parse_args()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    draw = mp.solutions.drawing_utils

    detect = GestureDetector()
    mouse = MouseHold()

    cap = cv2.VideoCapture(args.cam)

    # State machine: NONE, RMB_ONLY, BOTH
    state = "NONE"
    palm_count = fist_count = none_count = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = hands.process(rgb)

        h, w, _ = frame.shape
        palm = False
        fist = False
        hand_present = False

        if res.multi_hand_landmarks:
            hand_present = True
            hand = res.multi_hand_landmarks[0]
            draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            lms = hand.landmark
            palm = detect.is_open_palm(lms, w, h)
            fist = detect.is_fist(lms, w, h)

        # debounce counters
        palm_count = palm_count + 1 if palm else 0
        fist_count = fist_count + 1 if fist else 0
        none_count = none_count + 1 if not hand_present else 0

        # --- State machine ---
        if state == "BOTH":
            # Hold BOTH until palm opens (=> RMB only) or the hand disappears long enough
            if palm_count >= args.confirm_on:
                mouse.hold_left(False)
                mouse.hold_right(True)
                state = "RMB_ONLY"
                none_count = 0
            elif not hand_present and none_count >= args.none_for_release:
                mouse.hold_left(False)
                mouse.hold_right(False)
                state = "NONE"
            else:
                mouse.hold_left(True)
                mouse.hold_right(True)

        elif state == "RMB_ONLY":
            # Fist overrides to BOTH
            if fist_count >= args.confirm_on:
                mouse.hold_left(True)
                mouse.hold_right(True)
                state = "BOTH"
                none_count = 0
            elif not hand_present:
                if none_count >= args.none_for_release:
                    mouse.hold_right(False)
                    state = "NONE"
            else:
                # Keep RMB only while palm stays visible
                if palm:
                    mouse.hold_right(True)
                else:
                    mouse.hold_right(False)
                    state = "NONE"

        else:  # NONE
            if fist_count >= args.confirm_on:
                mouse.hold_left(True)
                mouse.hold_right(True)
                state = "BOTH"
                none_count = 0
            elif palm_count >= args.confirm_on:
                mouse.hold_right(True)
                state = "RMB_ONLY"
                none_count = 0
            else:
                mouse.hold_left(False)
                mouse.hold_right(False)

        # --- Overlay
        cv2.putText(frame, f"Palm={palm}  Fist={fist}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"State={state}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame,
                    f"LMB={'HOLD' if mouse.lmb_down else 'UP'}   RMB={'HOLD' if mouse.rmb_down else 'UP'}",
                    (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (0, 255, 0) if (mouse.lmb_down or mouse.rmb_down) else (0, 0, 255), 2)
        cv2.putText(frame, "Palm -> hold RMB | Fist -> hold LMB+RMB | Q to quit",
                    (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)

        cv2.imshow("Palm = RMB hold | Fist = LMB+RMB hold (cam=1)", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    mouse.hold_left(False)
    mouse.hold_right(False)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

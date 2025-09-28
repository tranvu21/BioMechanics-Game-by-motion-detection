
import argparse
import cv2
from pynput.keyboard import Listener, Key

from motion_modules import (
    KeyController, ActionLatch,
    PoseEstimator, HandsEstimator,
    Thresholds, MotionClassifier, GameMapper
)

def main():
    parser = argparse.ArgumentParser(description="MediaPipe Pose+Hands -> Latched Keyboard Controller")
    parser.add_argument("--buffer", type=int, default=20, help="Frames for speed smoothing")
    parser.add_argument("--idle_speed", type=float, default=120.0, help="Average speed below => Idle")
    parser.add_argument("--walk_speed", type=float, default=200.0, help="Average speed for Walking")
    parser.add_argument("--run_speed", type=float, default=300.0, help="Per-hand speed; BOTH over => Running")
    parser.add_argument("--yaw_thresh", type=float, default=0.30, help="Shoulder z-diff threshold for left/right")
    parser.add_argument("--raise_margin", type=float, default=0.05, help="Wrist must be this much above shoulder for 'raised'")
    parser.add_argument("--switch_frames", type=int, default=4, help="Consecutive frames needed to switch to a different action")
    parser.add_argument("--none_grace", type=int, default=8, help="Consecutive empty frames before releasing keys")
    parser.add_argument("--no_shift", action="store_true", help="Do not hold SHIFT while running")
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    # Modules
    kb = KeyController(use_shift=(not args.no_shift))
    pose = PoseEstimator(buffer=args.buffer)
    hands = HandsEstimator()
    thresholds = Thresholds(
        idle_speed=args.idle_speed,
        walk_speed=args.walk_speed,
        run_speed=args.run_speed,
        yaw_thresh=args.yaw_thresh,
        raise_margin=args.raise_margin,
    )
    clf = MotionClassifier(thresholds)
    mapper = GameMapper(kb)
    latch = ActionLatch(switch_frames=args.switch_frames, none_grace=args.none_grace)

    # State
    control_enabled = True
    running = True

    def on_key_press(key):
        nonlocal control_enabled, running
        if key == Key.f8:
            control_enabled = not control_enabled
            print(f"[TOGGLE] control_enabled = {control_enabled}")
            if not control_enabled:
                kb.release_all()
        elif key == Key.esc:
            print("[EXIT] ESC pressed")
            running = False
            return False

    listener = Listener(on_press=on_key_press); listener.start()

    cap = cv2.VideoCapture(args.cam)

    while cap.isOpened() and running:
        ok, frame = cap.read()
        if not ok:
            break

        # --- Pose pass
        landmarks, speeds, direction = pose.process(frame)

        # --- Classification
        # --- Gesture: both hands up (compute first so we can bypass movement)
        hands_up = clf.both_hands_up(landmarks)

        # --- Classification
        # Skip movement classification while hands_up so it won't flip to Walking/Running
        if hands_up:
            activity = "Jump (SPACE)"
        else:
            activity = clf.classify(speeds, direction)

        # --- Hands pass (collect)
        raised  = clf.wrist_raised(landmarks)
        closed  = hands.any_closed(frame)
        collect = raised and closed

        # --- Keys (raw) — Space has priority and suppresses movement
        if hands_up:
            raw_keys = {kb.SPACE_KEY}        # jump
        else:
            raw_keys = mapper.map_activity(activity)  # A/D/W or Shift+...

        # If you want Space to be exclusive of E, uncomment next line:
        # if hands_up: collect = False

        # Layer collect on top
        raw_keys = mapper.maybe_add_collect(raw_keys, collect)

        # --- Latch keys
        latched_keys = latch.update(raw_keys)

        # --- Apply keyboard
        if control_enabled:
            kb.apply(latched_keys)
        else:
            kb.release_all()
        # --- Overlay & skeleton
        yaw_label = clf.yaw_label(direction)
        raw_names = '+'.join([k.upper() if isinstance(k, str) else 'SHIFT' for k in raw_keys]) if raw_keys else '(none)'
        latched_names = '+'.join([k.upper() if isinstance(k, str) else 'SHIFT' for k in latched_keys]) if latched_keys else '(none)'

        cv2.putText(frame, f"{activity}", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
        cv2.putText(frame, f"RAW: {raw_names}", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160,200,255), 2)
        cv2.putText(frame, f"LATCHED: {latched_names}", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,220,160), 2)
        cv2.putText(frame, f"L:{speeds.left:.0f} R:{speeds.right:.0f} Avg:{speeds.avg:.0f}", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, f"Yaw:{yaw_label} dz={direction.dz:+.2f}  F8 toggle  ESC quit", (40, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2)

        pose.draw(frame)
        cv2.imshow("Game Controller (Latched Keyboard)", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Cleanup
    kb.release_all()
    cap.release()
    cv2.destroyAllWindows()
    listener.stop()

if __name__ == "__main__":
    main()

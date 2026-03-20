import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import math


class PoseClassifier:
    def __init__(
        self,
        model_path: str | Path = "pose.task",
        webcam_index: int = 0,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize the pose landmarker and store webcam configuration.
        """
        self.webcam_index = webcam_index

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"model not found at {model_path}."
            )

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def detect(self, image_bgr):
        """
        Run pose detection on a BGR image.
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb,
        )
        return self.detector.detect(mp_image)

    def _dist(self, a, b) -> float:
        """
        Compute 2D Euclidean distance between two landmarks.
        """
        return math.hypot(a.x - b.x, a.y - b.y)

    def detect_hands_raised(self, pose_position) -> bool:
        """
        Detect whether both wrists are above the nose.
        """
        if not pose_position:
            return False
        lm = pose_position[0]
        nose = lm[0]
        left_wrist = lm[15]
        right_wrist = lm[16]
        return (left_wrist.y < nose.y) and (right_wrist.y < nose.y)

    def detect_hands_together(self, pose_position) -> bool:
        """
        Detect whether wrists are close together relative to shoulder width.
        """
        if not pose_position:
            return False
        lm = pose_position[0]
        left_wrist = lm[15]
        right_wrist = lm[16]
        left_shoulder = lm[11]
        right_shoulder = lm[12]
        shoulder_width = max(self._dist(left_shoulder, right_shoulder), 1e-6)
        return self._dist(left_wrist, right_wrist) < 0.25 * shoulder_width

    def detect_arms_crossed(self, pose_position) -> bool:
        """
        Detect an "arms crossed" pose.
        """
        if not pose_position:
            return False
        lm = pose_position[0]
        left_wrist = lm[15]
        right_wrist = lm[16]
        left_shoulder = lm[11]
        right_shoulder = lm[12]
        shoulder_width = max(self._dist(left_shoulder, right_shoulder), 1e-6)

        left_to_right_shoulder = self._dist(left_wrist, right_shoulder)
        right_to_left_shoulder = self._dist(right_wrist, left_shoulder)
        wrists_close = self._dist(left_wrist, right_wrist) < 0.6 * shoulder_width

        return (
            left_to_right_shoulder < 0.55 * shoulder_width
            and right_to_left_shoulder < 0.55 * shoulder_width
            and wrists_close
        )

    def detect_hands_on_head(self, pose_position) -> bool:
        """
        Detect whether both hands are near the head (ears).
        """
        if not pose_position:
            return False
        lm = pose_position[0]
        left_wrist = lm[15]
        right_wrist = lm[16]
        left_ear = lm[7]
        right_ear = lm[8]
        left_shoulder = lm[11]
        right_shoulder = lm[12]
        shoulder_width = max(self._dist(left_shoulder, right_shoulder), 1e-6)

        left_near_head = min(self._dist(left_wrist, left_ear), self._dist(left_wrist, right_ear)) < 0.55 * shoulder_width
        right_near_head = min(self._dist(right_wrist, left_ear), self._dist(right_wrist, right_ear)) < 0.55 * shoulder_width
        return left_near_head and right_near_head

    def identify_match_pose(self, pose_position):
        """
        Classify the pose into a gesture label.
        """
        if not pose_position:
            return None

        if self.detect_hands_on_head(pose_position):
            return "hands_on_head"
        if self.detect_arms_crossed(pose_position):
            return "arms_crossed"
        if self.detect_hands_together(pose_position):
            return "hands_together"
        if self.detect_hands_raised(pose_position):
            return "hands_raised"

        return "neutral"

    def run(self):
        """
        Run real-time webcam pose classification.
        """
        cap = None
        if hasattr(cv2, "CAP_AVFOUNDATION"):
            cap = cv2.VideoCapture(self.webcam_index, cv2.CAP_AVFOUNDATION)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(self.webcam_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"could not open camera current: {self.webcam_index})."
            )

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                result = self.detect(frame)
                pose_position = result.pose_position

                label = self.identify_match_pose(pose_position)
                if label:
                    cv2.putText(
                        frame,
                        f"Action: {label}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow("Pose classifier", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

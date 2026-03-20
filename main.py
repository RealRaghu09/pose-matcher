from MonkeyPoseClassification import MonkeyPoseClassifier


def main():
    detector = MonkeyPoseClassifier(model_path="pose.task", webcam_index=0)
    detector.run()


if __name__ == "__main__":
    main()

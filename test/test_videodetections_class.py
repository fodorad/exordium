import unittest
import os
import numpy as np
from exordium import DATA_DIR
from exordium.video.detection import VideoDetections, FrameDetections, DetectionFromImage


class VideoDetectionsTestCase(unittest.TestCase):

    def setUp(self):
        self.video_detections = VideoDetections()
        self.detection1 = DetectionFromImage(frame_id=0, source='frame1.jpg', score=0.9, bb_xywh=np.zeros((4,)), landmarks=np.zeros((5,2)))
        self.detection2 = DetectionFromImage(frame_id=1, source='frame2.jpg', score=0.8, bb_xywh=np.zeros((4,)), landmarks=np.zeros((5,2)))
        self.detection3 = DetectionFromImage(frame_id=0, source='frame1.jpg', score=0.9, bb_xywh=np.zeros((4,)), landmarks=np.zeros((5,2)))
        self.detection4 = DetectionFromImage(frame_id=1, source='frame2.jpg', score=0.8, bb_xywh=np.zeros((4,)), landmarks=np.zeros((5,2)))
        self.detection5 = DetectionFromImage(frame_id=2, source='frame2.jpg', score=0.7, bb_xywh=np.zeros((4,)), landmarks=np.zeros((5,2)))
        self.detection6 = DetectionFromImage(frame_id=3, source='frame3.jpg', score=0.7, bb_xywh=np.zeros((4,)), landmarks=np.zeros((5,2)))

    def test_add(self):
        fdet1 = FrameDetections()
        fdet2 = FrameDetections()
        self.video_detections.add(fdet1.add(self.detection1))
        self.video_detections.add(fdet2.add(self.detection2))
        self.assertEqual(len(self.video_detections), 2)
        self.assertEqual(self.video_detections[0], fdet1)
        self.assertEqual(self.video_detections[1], fdet2)

    def test_add_empty(self):
        fdet1 = FrameDetections()
        fdet2 = FrameDetections()
        self.video_detections.add(fdet1)
        self.video_detections.add(fdet2)
        self.assertEqual(len(self.video_detections), 0)

    def test_iterator(self):
        fdet1 = FrameDetections()
        fdet2 = FrameDetections()
        self.video_detections.add(fdet1.add(self.detection1))
        self.video_detections.add(fdet2.add(self.detection2))
        for idx, frame_detections in enumerate(self.video_detections):
            self.assertEqual(frame_detections, self.video_detections[idx])

    def test_equal(self):
        # instance 1
        fdet1 = FrameDetections()
        fdet2 = FrameDetections()
        self.video_detections.add(fdet1.add(self.detection1))
        self.video_detections.add(fdet2.add(self.detection2))
        # instance 2
        other_video_detections1 = VideoDetections()
        fdet3 = FrameDetections()
        fdet4 = FrameDetections()
        other_video_detections1.add(fdet3.add(self.detection3))
        other_video_detections1.add(fdet4.add(self.detection4))
        # instance 3
        other_video_detections2 = VideoDetections()
        fdet5 = FrameDetections()
        fdet6 = FrameDetections()
        other_video_detections2.add(fdet5.add(self.detection5))
        other_video_detections2.add(fdet6.add(self.detection6))

        self.assertEqual(self.video_detections, other_video_detections1)
        self.assertNotEqual(self.video_detections, other_video_detections2)
        self.assertNotEqual(other_video_detections1, other_video_detections2)

    def test_save_and_load(self):
        detection1 = DetectionFromImage(frame_id=0, source='000000.png', score=0.9, bb_xywh=np.array([10, 20, 30, 40]), landmarks=np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]))
        detection2 = DetectionFromImage(frame_id=1, source='000001.png', score=0.8, bb_xywh=np.array([15, 25, 35, 45]), landmarks=np.array([[11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]))
        detection3 = DetectionFromImage(frame_id=1, source='000001.png', score=0.7, bb_xywh=np.array([30, 40, 20, 20]), landmarks=np.array([[11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]))
        fdet1 = FrameDetections()
        fdet2 = FrameDetections()
        self.video_detections.add(fdet1.add(detection1))
        self.video_detections.add(fdet2.add(detection2).add(detection3))
        output_file = str(DATA_DIR / 'output.csv')
        self.video_detections.save(output_file)
        loaded_video_detections = VideoDetections().load(output_file)
        self.assertEqual(self.video_detections, loaded_video_detections)
        os.remove(output_file)


if __name__ == '__main__':
    unittest.main()
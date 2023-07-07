import unittest
import os
import numpy as np
from exordium.video.detection import VideoDetections, FrameDetections, Detection


class VideoDetectionsTestCase(unittest.TestCase):

    def setUp(self):
        self.video_detections = VideoDetections()

    def test_add(self):
        fdet1 = FrameDetections()
        detection1 = Detection(0, 'frame1.jpg', 0.9, np.array([10, 20, 30, 40]), np.array([10, 20, 40, 60]), np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]))
        fdet1.add_detection(detection1)
        fdet2 = FrameDetections()
        detection2 = Detection(1, 'frame2.jpg', 0.8, np.array([15, 25, 35, 45]), np.array([15, 25, 50, 70]), np.array([[11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]))
        fdet2.add_detection(detection2)
        self.video_detections.add(fdet1)
        self.video_detections.add(fdet2)
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
        detection1 = Detection(0, 'frame1.jpg', 0.9, np.array([10, 20, 30, 40]), np.array([10, 20, 40, 60]), np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]))
        fdet1.add_detection(detection1)
        fdet2 = FrameDetections()
        detection2 = Detection(1, 'frame2.jpg', 0.8, np.array([15, 25, 35, 45]), np.array([15, 25, 50, 70]), np.array([[11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]))
        fdet2.add_detection(detection2)
        self.video_detections.add(fdet1)
        self.video_detections.add(fdet2)

        for idx, frame_detections in enumerate(self.video_detections):
            self.assertEqual(frame_detections, self.video_detections[idx])

    def test_equal(self):
        fdet1 = FrameDetections()
        detection1 = Detection(0, 'frame1.jpg', 0.9, np.array([10, 20, 30, 40]), np.array([10, 20, 40, 60]), np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]))
        fdet1.add_detection(detection1)
        fdet2 = FrameDetections()
        detection2 = Detection(1, 'frame2.jpg', 0.8, np.array([15, 25, 35, 45]), np.array([15, 25, 50, 70]), np.array([[11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]))
        fdet2.add_detection(detection2)
        self.video_detections.add(fdet1)
        self.video_detections.add(fdet2)

        fdet1 = FrameDetections()
        detection1 = Detection(0, 'frame1.jpg', 0.9, np.array([10, 20, 30, 40]), np.array([10, 20, 40, 60]), np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]))
        fdet1.add_detection(detection1)
        fdet2 = FrameDetections()
        detection2 = Detection(1, 'frame2.jpg', 0.8, np.array([15, 25, 35, 45]), np.array([15, 25, 50, 70]), np.array([[11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]))
        fdet2.add_detection(detection2)
        other_video_detections1 = VideoDetections()
        other_video_detections1.add(fdet1)
        other_video_detections1.add(fdet2)

        fdet1 = FrameDetections()
        detection1 = Detection(2, 'frame2.jpg', 0.7, np.array([10, 20, 30, 40]), np.array([10, 20, 40, 60]), np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]))
        fdet1.add_detection(detection1)
        fdet2 = FrameDetections()
        detection2 = Detection(3, 'frame3.jpg', 0.7, np.array([15, 25, 35, 45]), np.array([15, 25, 50, 70]), np.array([[11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]))
        fdet2.add_detection(detection2)
        other_video_detections2 = VideoDetections()
        other_video_detections2.add(fdet1)
        other_video_detections2.add(fdet2)

        self.assertEqual(self.video_detections, other_video_detections1)
        self.assertNotEqual(self.video_detections, other_video_detections2)
        self.assertNotEqual(other_video_detections1, other_video_detections2)

    def test_save_and_load(self):
        fdet1 = FrameDetections()
        detection1 = Detection(0, '000000.png', 0.9, np.array([10, 20, 30, 40]), np.array([10, 20, 40, 60]), np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]))
        fdet1.add_detection(detection1)

        fdet2 = FrameDetections()
        detection2 = Detection(1, '000001.png', 0.8, np.array([15, 25, 35, 45]), np.array([15, 25, 50, 70]), np.array([[11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]))
        detection3 = Detection(1, '000001.png', 0.7, np.array([30, 40, 20, 20]), np.array([30, 40, 50, 60]), np.array([[11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]))
        fdet2.add_detection(detection2)
        fdet2.add_detection(detection3)

        self.video_detections.add(fdet1)
        self.video_detections.add(fdet2)

        output_file = 'output.csv'
        self.video_detections.save(output_file)
        loaded_video_detections = VideoDetections().load(output_file)

        self.assertEqual(self.video_detections, loaded_video_detections)

        os.remove(output_file)        


if __name__ == '__main__':
    unittest.main()
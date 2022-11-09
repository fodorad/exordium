import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Print(nn.Module):
    def __init__(self, description=None):
        self.description = description
        super(Print, self).__init__()

    def forward(self, x):
        if not self.description is None:
            print(self.description)
        print(x.shape)
        return x


class IrisBlock(nn.Module):
    """This is the main building block for architecture"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super(IrisBlock, self).__init__()
        

        # My impl
        self.stride = stride
        self.channel_pad = out_channels - in_channels
        
        padding = (kernel_size - 1) // 2
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)

        self.convAct = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=int(out_channels/2), kernel_size=stride, stride=stride, padding=0, bias=True),
            nn.PReLU(int(out_channels/2))
        )
        self.dwConvConv = nn.Sequential(
            nn.Conv2d(in_channels=int(out_channels/2), out_channels=int(out_channels/2), 
                      kernel_size=kernel_size, stride=1, padding=padding,  # Padding might be wrong here
                      groups=int(out_channels/2), bias=True),
            nn.Conv2d(in_channels=int(out_channels/2), out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.act = nn.PReLU(out_channels)

    def forward(self, x):
        h = self.convAct(x)
        if self.stride == 2:
            
            x = self.max_pool(x)
        
        h = self.dwConvConv(h)

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)
        
        return self.act(h + x)


class IrisLandmarks(nn.Module):
    """The IrisLandmark face landmark model from MediaPipe.
    Because we won't be training this model, it doesn't need to have
    batchnorm layers. These have already been "folded" into the conv 
    weights by TFLite.
    The conversion to PyTorch is fairly straightforward, but there are 
    some small differences between TFLite and PyTorch in how they handle
    padding on conv layers with stride 2.
    This version works on batches, while the MediaPipe version can only
    handle a single image at a time.
    """
    def __init__(self):
        super(IrisLandmarks, self).__init__()

        # self.num_coords = 228
        # self.x_scale = 64.0
        # self.y_scale = 64.0
        self.min_score_thresh = 0.75

        self._define_layers()

    def _define_layers(self):
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=0, bias=True),
            nn.PReLU(64),

            IrisBlock(64, 64),
            IrisBlock(64, 64),
            IrisBlock(64, 64),
            IrisBlock(64, 64),
            IrisBlock(64, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2)
        )
        self.split_eye = nn.Sequential(
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            nn.Conv2d(in_channels=128, out_channels=213, kernel_size=2, stride=1, padding=0, bias=True)
        )
        self.split_iris = nn.Sequential(
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            nn.Conv2d(in_channels=128, out_channels=15, kernel_size=2, stride=1, padding=0, bias=True)
        )

        
        
    def forward(self, x):
        # TFLite uses slightly different padding on the first conv layer
        # than PyTorch, so do it manually.
        x = F.pad(x, [0, 1, 0, 1], "constant", 0)
        b = x.shape[0]      # batch size, needed for reshaping later

        x = self.backbone(x)            # (b, 128, 8, 8)
        
        e = self.split_eye(x)           # (b, 213, 1, 1)    
        e = e.view(b, -1)               # (b, 213)
        
        i = self.split_iris(x)          # (b, 15, 1, 1)
        i = i.reshape(b, -1)            # (b, 15)
        
        return [e, i]

    def _device(self):
        """Which device (CPU or GPU) is being used by this model?"""
        return self.backbone[0].weight.device
    
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()        
    
    def _preprocess(self, x):
        """Converts the image pixels to the range [-1, 1]."""
        # return x.float() / 127.5 - 1.0
        return x.float() / 255.0 # NOTE: [0.0, 1.0] range seems to give better results

    def predict_on_image(self, img):
        """Makes a prediction on a single image.
        Arguments:
            img: a NumPy array of shape (H, W, 3) or a PyTorch tensor of
                 shape (3, H, W). The image's height and width should be 
                 64 pixels.
        Returns:
            A tensor with face detections.
        """
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute((2, 0, 1))
        
        return self.predict_on_batch(img.unsqueeze(0))

    def predict_on_batch(self, x):
        """Makes a prediction on a batch of images.
        Arguments:
            x: a NumPy array of shape (b, H, W, 3) or a PyTorch tensor of
               shape (b, 3, H, W). The height and width should be 64 pixels.
        Returns:
            A list containing a tensor of face detections for each image in 
            the batch. If no faces are found for an image, returns a tensor
            of shape (0, 17).
        Each face detection is a PyTorch tensor consisting of 17 numbers:
            - ymin, xmin, ymax, xmax
            - x,y-coordinates for the 6 keypoints
            - confidence score
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).permute((0, 3, 1, 2))
            # x = torch.from_numpy(x)

        assert x.shape[1] == 3
        assert x.shape[2] == 64
        assert x.shape[3] == 64

        # 1. Preprocess the images into tensors:
        x = x.to(self._device())
        x = self._preprocess(x)

        # 2. Run the neural network:
        with torch.no_grad():
            out = self.__call__(x)

        # 3. Postprocess the raw predictions:
        eye, iris = out

        return eye.view(-1, 71, 3), iris.view(-1, 5, 3)


class IrisPredictor():

    def __init__(self, gpu: str = 'cuda:0'):
        self.net = IrisLandmarks().to(gpu)
        self.net.load_weights(f"{os.path.dirname(__file__)}/irislandmarks.pth")

    def predict(self, img):
        eye_gpu, iris_gpu = self.net.predict_on_image(img)
        eye = eye_gpu.cpu().numpy().squeeze(0)[:,:2] # (71, 2)
        iris = iris_gpu.cpu().numpy().squeeze(0)[:,:2] # (5, 2)
        return eye, iris


'''
    2
3   0   1
    4   
'''
IRIS_LANDMARKS = {
    'center': 0,
    'left': 3,
    'top': 2,
    'right': 1,
    'bottom': 4
}

# TODO
def get_eye(img, point, bb_size: int = 64):
    return img[int(point[1]-bb_size//2):int(point[1]+bb_size//2),
               int(point[0]-bb_size//2):int(point[0]+bb_size//2), :]

def crop_eye_pipeline(img_path):

    import cv2
    from exordium.video.face import face_crop_with_landmarks
    from exordium.video.tddfa_v2 import FACE_LANDMARKS

    d = face_crop_with_landmarks(img_path)
    d = d[0]
    xy_min = np.min(d['landmarks'][np.array(FACE_LANDMARKS['left_eye']),:], axis=0) # (2,)
    xy_max = np.max(d['landmarks'][np.array(FACE_LANDMARKS['right_eye']),:], axis=0) # (2,)

    img = d['img']


if __name__ == "__main__":

    import cv2
    import matplotlib
    #matplotlib.use('QtAgg')
    import matplotlib.pyplot as plt
    from exordium.video.face import face_crop_with_landmarks
    from exordium.video.tddfa_v2 import FACE_LANDMARKS
 
    img_path = 'data/processed/frames/h-jMFLm6U_Y.000/frame_00001.png'
    img_path = '/home/fodor/dev/eye/data/talkingFace/frames/000001.png'
    d = face_crop_with_landmarks(img_path)
    d = d[0]
    print(d['img'].shape)
    print(d['landmarks'].shape)
    print()
    le = np.mean(d['landmarks'][np.array(FACE_LANDMARKS['left_eye']),:], axis=0)
    re = np.mean(d['landmarks'][np.array(FACE_LANDMARKS['right_eye']),:], axis=0)
    
    plt.figure()
    plt.imshow(d['img'][:,:,::-1])
    plt.scatter(le[0], le[1], s=5.0, color='b')
    plt.scatter(re[0], re[1], s=5.0, color='g')
    plt.savefig('iris_out.png')

    le_eye = get_eye(d['img'], le)
    re_eye = get_eye(d['img'], re)
    cv2.imwrite('left_eye.png', le_eye)
    cv2.imwrite('right_eye.png', re_eye)
    
    img = cv2.imread("data/processed/eye/test_eye.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64))

    ip = IrisPredictor()
    eye, iris = ip.predict(img)

    print(le_eye.shape)
    print(re_eye.shape)
    le_eye_out, le_iris_out = ip.predict(le_eye)
    re_eye_out, re_iris_out = ip.predict(re_eye)


    print('eye:', eye.shape)
    print('iris:', iris.shape)

    le_eye = cv2.cvtColor(le_eye, cv2.COLOR_BGR2RGB)
    re_eye = cv2.cvtColor(re_eye, cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(img)
    x, y = eye[:, 0], eye[:, 1]
    plt.scatter(x, y, s=5.0, color='b')


    x, y = iris[:, 0], iris[:, 1]
    plt.scatter(x, y, s=5.0, color='r')
    
    x, y = iris[4, 0], iris[4, 1]
    plt.scatter(x, y, s=5.0, color='g')
    plt.savefig('test_eye_out.jpg')



    plt.figure()
    plt.imshow(le_eye)
    x, y = le_eye_out[:, 0], le_eye_out[:, 1]
    plt.scatter(x, y, s=5.0, color='b')
    x, y = le_iris_out[:, 0], le_iris_out[:, 1]
    plt.scatter(x, y, s=5.0, color='r')
    plt.savefig('test_le_eye_out.jpg')
    plt.figure()
    plt.imshow(re_eye)
    x, y = re_eye_out[:, 0], re_eye_out[:, 1]
    plt.scatter(x, y, s=5.0, color='b')
    x, y = re_iris_out[:, 0], re_iris_out[:, 1]
    plt.scatter(x, y, s=5.0, color='r')
    plt.savefig('test_re_eye_out.jpg')

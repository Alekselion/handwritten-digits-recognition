import os
import cv2
import numpy as np
import torch
from torch import nn


# create model
class Net(nn.Module):
    def __init__(self, hid1, hid2):
        super(Net, self).__init__()
        
        # image 1*28*28
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # image 32*14*14
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # image 64*7*7
        self.fc_block = nn.Sequential(
            nn.Linear(64*7*7, hid1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hid1, hid2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hid2, 10),
            nn.Softmax(dim=1)  
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc_block(x)
                    
        return x


model = Net(100, 80)

# load model
pth = os.path.join(os.getcwd(), "models")
model.load_state_dict(torch.load(pth + "/model_cnn.pth"))

# canvas
w, h = 300, 300
canvas = np.zeros((h, w, 3), np.uint8)  # create canvas
title = np.zeros((30, w, 3), np.uint8)  # create title
canvas[:, :, :] = (255, 255, 255)  # change color for canvas
title[:, :, :] = (210, 210, 210)  # change color for title
flag = False


def drawCircle(event, x, y, flags, param):
    """ left mouse button - draw on canvas
        right mouse button - clear canvas
    """
    global flag
    if event == cv2.EVENT_LBUTTONDOWN:
        flag = True
        cv2.circle(canvas, (x,y), 15, (0, 0, 0), -1)
        title[:, :, :] = (210, 210, 210)
    elif event == cv2.EVENT_LBUTTONUP:
        flag = False

    # draw
    if flag and event == cv2.EVENT_MOUSEMOVE:
        cv2.circle(canvas, (x,y), 15, (0, 0, 0), -1)
        title[:, :, :] = (210, 210, 210)

    # clear
    if not flag and event == cv2.EVENT_RBUTTONDOWN:
        canvas[:, :, :] = (255, 255, 255)
        title[:, :, :] = (210, 210, 210)


cv2.namedWindow("canvas")  
cv2.setMouseCallback("canvas", drawCircle)
while True:
    # image processing
    img = canvas[30:, :, :].copy()  # copy canvas without title
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # change image color to gray
    img = cv2.resize(img, (28, 28))  # image resize to 28*28
    img = torch.from_numpy(img)  # image to tensor
    img = 255.0 - img  # image inversion
    img = img / 255.0  # image normalize  img.shape = (W, H)
    img = img.unsqueeze(0)  # add 'batch size'  img.shape = (B, W, H)
    img = img.unsqueeze(1)  # add 'channel'  img.shape = (B, C, W, H)
    
     # predict
    model.eval()
    pred = (model.forward(img))
    predNumber = int(pred.argmax(dim=1))
    predAccuracy = float(pred[0][predNumber]) * 100
    result = f"{predNumber} ({predAccuracy :.2f}%)"
    
    # show
    cv2.putText(title, "Number:", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    cv2.putText(title, result, (85, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    canvas[:30, :] = title
    cv2.imshow("canvas", canvas)
    if cv2.waitKey(1) & 0xFF == ord("q"):  
        break

cv2.destroyAllWindows()
import os
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# create model
class Net(nn.Module):
    def __init__(self, hid1, hid2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, hid1)
        self.fc2 = nn.Linear(hid1, hid2)
        self.fc3 = nn.Linear(hid2, 10)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28*28)  # flatten
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.softmax(self.fc3(x), dim=1)
        return x


model = Net(100, 80)

# load model
pth = os.path.join(os.getcwd(), "models")
model.load_state_dict(torch.load(pth + "/model_ffnn.pth"))

# canvas
w, h = 300, 300
canvas = np.zeros((h, w, 3), np.uint8)  # create canvas
title = np.zeros((30, w, 3), np.uint8)  # create title
title[:, :, :] = (50, 50, 50)  # change color for title
flag = False


def drawCircle(event, x, y, flags, param):
    """ left mouse button - draw on canvas
        right mouse button - clear canvas
    """
    global flag
    if event == cv2.EVENT_LBUTTONDOWN:
        flag = True
        cv2.circle(canvas, (x,y), 15, (255, 255, 255), -1)
        title[:, :, :] = (50, 50, 50)
    elif event == cv2.EVENT_LBUTTONUP:
        flag = False

    # draw
    if flag and event == cv2.EVENT_MOUSEMOVE:
        cv2.circle(canvas, (x,y), 15, (255, 255, 255), -1)
        title[:, :, :] = (50, 50, 50)

    # clear
    if not flag and event == cv2.EVENT_RBUTTONDOWN:
        canvas[:, :, :] = (0, 0, 0)
        title[:, :, :] = (50, 50, 50)


cv2.namedWindow("canvas")  
cv2.setMouseCallback("canvas", drawCircle)
while True:
    # image processing
    img = canvas[30:, :, :].copy()  # copy canvas without title
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # change image color to gray
    img = cv2.resize(img, (28, 28))  # image resize to 28*28
    img = torch.from_numpy(img)  # image to tensor
    img = img / 255.0  # image normalize
    
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
import os
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

colorText = (255, 0, 255)

# ############
# SELECT MODEL
# ############
 
# create canvas for model selection
canvas = np.zeros((300, 300, 3), np.uint8)
canvas[:, 150:, :] = (255, 255, 255)
cv2.putText(canvas, "Click on model", (60, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, colorText, 2)
cv2.putText(canvas, "to select it", (85, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, colorText, 2)
cv2.putText(canvas, "Convolutional", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colorText, 2)
cv2.putText(canvas, "Neural Network", (3, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colorText, 2)
cv2.putText(canvas, "Feed Forward", (160, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colorText, 2)
cv2.putText(canvas, "Neural Network", (153, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colorText, 2)
modelName = None


def selectModel(event, x, y, *otherParams):
    if event == cv2.EVENT_LBUTTONDOWN:
        global modelName
        modelName = "cnn" if x <= 150 else "ffnn"


cv2.namedWindow("Select model")  
cv2.setMouseCallback("Select model", selectModel)
while True:
    cv2.imshow("Select model", canvas)
    if modelName: break
    cv2.waitKey(1)
cv2.destroyWindow("Select model")

# ############
# SELECT COLOR
# ############

# create canvas for color selection
canvas = np.zeros((300, 300, 3), np.uint8)
canvas[:, 150:, :] = (255, 255, 255)
cv2.putText(canvas, "Click on color", (60, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, colorText, 2)
cv2.putText(canvas, "to select it", (85, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, colorText, 2)
cv2.putText(canvas, "Black", (45, 135), cv2.FONT_HERSHEY_COMPLEX, 0.7, colorText, 2)
cv2.putText(canvas, "Canvas", (35, 165), cv2.FONT_HERSHEY_COMPLEX, 0.7, colorText, 2)
cv2.putText(canvas, "White", (195, 135), cv2.FONT_HERSHEY_COMPLEX, 0.7, colorText, 2)
cv2.putText(canvas, "Canvas", (185, 165), cv2.FONT_HERSHEY_COMPLEX, 0.7, colorText, 2)
colorName = None


def selectColor(event, x, y, *otherParams):
    if event == cv2.EVENT_LBUTTONDOWN:
        global colorName
        colorName = "black" if x <= 150 else "white"


cv2.namedWindow("Select color")  
cv2.setMouseCallback("Select color", selectColor)
while True:
    cv2.imshow("Select color", canvas)
    if colorName: break
    cv2.waitKey(1)
cv2.destroyWindow("Select color")

# ##########
# LOAD MODEL
# ##########

if modelName == "ffnn":
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
    pth = os.path.join(os.getcwd(), "models")
    model.load_state_dict(torch.load(pth + "/model_ffnn.pth"))

elif modelName == "cnn":
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
    pth = os.path.join(os.getcwd(), "models")
    model.load_state_dict(torch.load(pth + "/model_cnn.pth"))

# #############
# CREATE CANVAS
# #############

w, h = 300, 400
canvas = np.zeros((h, w, 3), np.uint8)  # create canvas
header = np.zeros((30, w, 3), np.uint8)  # create header
footer = np.zeros((60, w, 3), np.uint8)  # create footer

# initialize color
canvasColor = (255, 255, 255) if colorName == "white" else (0, 0, 0)  # color for canvas
headerColor = (205, 205, 205) if colorName == "white" else (50, 50, 50)  # color for header
footerColor = (205, 205, 205) if colorName == "white" else (50, 50, 50)  # color for footer
penColor = (0, 0, 0) if colorName == "white" else (255, 255, 255)  # color for painting

# change color
canvas[:, :, :] = canvasColor  # color for canvas
header[:, :, :] = headerColor  # color for header
footer[:, :, :] = footerColor  # color for footer
cv2.putText(footer, "Left mouse button to draw", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colorText, 2)
cv2.putText(footer, "Right mouse button to clear", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colorText, 2)
canvas[340:, :, :] = footer
flag = False


def drawCircle(event, x, y, flags, param):
    """ left mouse button - draw on canvas
        right mouse button - clear canvas
    """
    global flag
    if event == cv2.EVENT_LBUTTONDOWN:
        flag = True
        cv2.circle(canvas, (x,y), 15, penColor, -1)
        header[:, :, :] = headerColor
    elif event == cv2.EVENT_LBUTTONUP:
        flag = False

    # draw
    if flag and event == cv2.EVENT_MOUSEMOVE:
        cv2.circle(canvas, (x,y), 15, penColor, -1)
        header[:, :, :] = headerColor

    # clear
    if not flag and event == cv2.EVENT_RBUTTONDOWN:
        canvas[:340, :, :] = canvasColor
        header[:, :, :] = headerColor


cv2.namedWindow("canvas")  
cv2.setMouseCallback("canvas", drawCircle)
while True:
    # image processing
    img = canvas[30:340, :, :].copy()  # copy canvas without header and footer
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # change image color to gray
    img = cv2.resize(img, (28, 28))  # image resize to 28*28
    img = torch.from_numpy(img)  # image to tensor
    if colorName == "white":
        img = 255.0 - img  # image inversion
    
    img = img / 255.0  # image normalize
    if modelName == "cnn":
        img = img.unsqueeze(0)  # add 'batch size'  img.shape = (B, W, H)
        img = img.unsqueeze(1)  # add 'channel'  img.shape = (B, C, W, H)
    
    # predict
    model.eval()
    pred = (model.forward(img))
    predNumber = int(pred.argmax(dim=1))
    predAccuracy = float(pred[0][predNumber]) * 100
    result = f"{predNumber} ({predAccuracy :.2f}%)"
    
    # show
    cv2.putText(header, "Number:", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colorText, 2)
    cv2.putText(header, result, (95, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colorText, 2)
    canvas[:30, :, :] = header
    cv2.imshow("canvas", canvas)
    if cv2.waitKey(1) & 0xFF == ord("q"):  
        break

cv2.destroyAllWindows()
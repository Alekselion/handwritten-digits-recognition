import os
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# ###########
# PRE-SETTING 
# ###########

# color
textColor = (255, 0, 255)  # text color (purple)
contentColor = (0, 0, 0)  # canvas color (black)
fieldsColor = (50, 50, 50)  # header and footer color (gray)

# size
widthCanvas, heightCanvas = 300, 400  # canvas size
widthFields, heightFields = widthCanvas, 50  # header and footer size
widthContent, heightContent = widthCanvas, heightCanvas - heightFields * 2  # content size

# canvas
canvas = np.zeros((heightCanvas, widthCanvas, 3), np.uint8)  # create

# header
header = np.zeros((heightFields, widthFields, 3), np.uint8)  # create
header[:, :, :] = fieldsColor  # change color
canvas[:heightFields, :, :] = header  # add to canvas

# content
content = np.zeros((heightContent, widthFields, 3), np.uint8)  # create
content[:, :, :] = contentColor  # change color
canvas[heightFields:heightCanvas-heightFields, :, :] = content  # add to canvas

# footer
footer = np.zeros((heightFields, widthFields, 3), np.uint8)  # create
footer[:, :, :] = fieldsColor  # change color
canvas[heightCanvas-heightFields:, :, :] = footer  # add to canvas

# # show canvas
# cv2.imshow("Sample canvas", canvas)
# cv2.waitKey(0)
# cv2.destroyWindow("Sample canvas")

# ############
# SELECT MODEL
# ############

# change header
cv2.putText(header, "Click on model to select it", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2)
canvas[:heightFields, :, :] = header  # add header to canvas
header[:, :, :] = fieldsColor  # clear header

# change content
content[:, widthContent//2:, :] = (255, 255, 255)  # split content vertically
cv2.putText(content, "Convolutional", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2)
cv2.putText(content, "Neural Network", (3, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2)
cv2.putText(content, "Feed Forward", (160, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2)
cv2.putText(content, "Neural Network", (153, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2)
canvas[heightFields:heightCanvas-heightFields, :, :] = content  # add content to canvas
content[:, :, :] = contentColor  # clear content

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

# ############
# SELECT COLOR
# ############

# change header
cv2.putText(header, "Click on color to select it", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2)
canvas[:heightFields, :, :] = header  # add header to canvas
header[:, :, :] = fieldsColor  # clear header

# change content
content[:, widthContent//2:, :] = (255, 255, 255)  # split content vertically
cv2.putText(content, "Black canvas", (15, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2)
cv2.putText(content, "Background", (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2)
cv2.putText(content, "White canvas", (165, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2)
cv2.putText(content, "Background", (170, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2)
canvas[heightFields:heightCanvas-heightFields, :, :] = content  # add content to canvas
content[:, :, :] = contentColor  # clear content

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
# SET COLORS
# ##########

contentColor = (255, 255, 255) if colorName == "white" else (0, 0, 0)  # color for content
fieldsColor = (205, 205, 205) if colorName == "white" else (50, 50, 50)  # color for header and footer
penColor = (0, 0, 0) if colorName == "white" else (255, 255, 255)  # color for painting

# change color
header[:, :, :] = fieldsColor  # color for header
canvas[:heightFields, :, :] = header  # add to canvas
content[:, :, :] = contentColor  # color for content
canvas[heightFields:heightCanvas-heightFields, :, :] = content  # add to canvas
footer[:, :, :] = fieldsColor  # color for footer
canvas[heightCanvas-heightFields:, :, :] = footer  # add to canvas

# # show canvas
# cv2.imshow("Sample canvas", canvas)
# cv2.waitKey(0)
# cv2.destroyWindow("Sample canvas")

# ########
# RUN TEST
# ########

# change footer
cv2.putText(footer, "Left mouse button to draw", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2)
cv2.putText(footer, "Right mouse button to clear", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2)
canvas[heightCanvas-heightFields:, :, :] = footer  # add to canvas
footer[:, :, :] = fieldsColor  # clear footer

flag = False


def drawCircle(event, x, y, *otherParams):
    """ left mouse button to draw
        right mouse button to clear
    """
    global flag
    if event == cv2.EVENT_LBUTTONDOWN:
        flag = True
        cv2.circle(content, (x, y-heightFields), 15, penColor, -1)
        canvas[heightFields:heightCanvas-heightFields, :, :] = content  # add to canvas
    elif event == cv2.EVENT_LBUTTONUP:
        flag = False

    # draw
    if flag and event == cv2.EVENT_MOUSEMOVE:
        cv2.circle(content, (x, y-heightFields), 15, penColor, -1)
        canvas[heightFields:heightCanvas-heightFields, :, :] = content  # add to canvas

    # clear
    if not flag and event == cv2.EVENT_RBUTTONDOWN:
        content[:, :, :] = contentColor  # clear content
        canvas[heightFields:heightCanvas-heightFields, :, :] = content  # add to canvas


emptyContent = content.copy()
cv2.namedWindow("canvas")  
cv2.setMouseCallback("canvas", drawCircle)
while True: 
    img = content.copy()  # copy content
    isEmpty = img.all() == emptyContent.all() if colorName == "white" else img.any() == emptyContent.any()
    if isEmpty:  # check content is empty
        # change header
        cv2.putText(header, "Canvas is empty", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2)
        canvas[:heightFields, :, :] = header  # add header to canvas
        header[:, :, :] = fieldsColor  # clear header
    else:
        # image processing
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
        result = f"Number: {predNumber} ({predAccuracy :.2f}%)"
        
        # change header
        cv2.putText(header, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2)
        canvas[:heightFields, :, :] = header  # add header to canvas
        header[:, :, :] = fieldsColor  # clear header
    
    cv2.imshow("canvas", canvas)
    if cv2.waitKey(1) & 0xFF == ord("q"):  
        break

cv2.destroyAllWindows()
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
contentColor = (255, 255, 255)  # canvas color (black)
fieldsColor = (205, 205, 205)  # header and footer color (gray)

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


# ############
# CREATE MODEL
# ############

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


model = Net(100, 80)  # create model
pth = os.path.join(os.getcwd(), "models")
model.load_state_dict(torch.load(pth + "/model_ffnn.pth"))  # load model weights

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
        cv2.circle(content, (x, y-heightFields), 15, (0, 0, 0), -1)
        canvas[heightFields:heightCanvas-heightFields, :, :] = content  # add to canvas
    elif event == cv2.EVENT_LBUTTONUP:
        flag = False

    # draw
    if flag and event == cv2.EVENT_MOUSEMOVE:
        cv2.circle(content, (x, y-heightFields), 15, (0, 0, 0), -1)
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
    if img.all() == emptyContent.all():  # check content is empty
        # change header
        cv2.putText(header, "Canvas is empty", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2)
        canvas[:heightFields, :, :] = header  # add header to canvas
        header[:, :, :] = fieldsColor  # clear header
    else:
        # image processing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # change image color to gray
        img = cv2.resize(img, (28, 28))  # image resize to 28*28
        img = torch.from_numpy(img)  # image to tensor
        img = 255.0 - img  # image inversion        
        img = img / 255.0  # image normalize

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
########################################################################
#
# File:   lab2.py
# Author: Emanuel Schorsch and Katie McMenamin
# Date:   March, 2015
#
# Written for ENGR 27 - Computer Vision - Project 2
#
########################################################################
#
# This project demonstrates a computer referee for two player tic-tac-toe
# Project the computer sceen and aim the webcam at the projected image
# so that the board is appropriately captured
#
# After an approx 30 sec setup phase the program will detect illegal moves
# and allow two human players to play one full game, keeping track of turns

# Usage: the program takes no command arguments. It always tries to capture 
# from device 0.

import cv2
import cv
import numpy as np
import sys
import struct

import findattributes
import matplotlib.pyplot as plt

# Tell python where to find cvk2 module before importing it.
sys.path.append('../cvk2')
import cvk2

"""
Return the centroid of the largest contour and the area
"""
def getDot(contours):
    #body represents the area of the largest contour and head of the 2nd largest
    body = -1
    head = -1
    dot = -1

    # Go through each contour and find the largest and 2nd largest contours
    for cnt in contours[0]:
        info = cv2.moments(cnt)
        area = info['m00']
        if area > head:
            if area > body:
                body = area
                dot = cnt
            else:
                head = area

    if body < 1:
        return (1,1), 1
    else:
        return cvk2.getcontourinfo(dot)['mean'], body


def subtractBackground(frame, background, threshold=120):
    orig_float = frame.astype(float)

    # For each pixel in the original image, subtract the temporal avg.
    dists_float = orig_float - background

    # Square the differences.
    dists_float = dists_float*dists_float

    # Sum across RGB to get one number per pixel. The result is an array
    dists_float = dists_float.sum(axis=2)

    # Take the square root to get a true distance in RGB space.
    dists_float = np.sqrt(dists_float)

    # Allocate space to convert back to uint8, and convert back.
    # This is better than writing
    # 
    #   dists_uint8 = dists_float.astype('uint8')
    #
    # Because it correctly handles overflow (values above 255).
    dists_uint8 = np.empty(dists_float.shape, 'uint8')
    cv2.convertScaleAbs(dists_float, dists_uint8, 1, 0)

    mask = np.zeros(dists_float.shape, 'uint8')

    # Create a mask by thresholding the distance image at <threshold>.  All pixels
    # with value less than <threshold> go to 255, and all pixels with value
    # greater than or equal to <threshold> go to 255.
    cv2.threshold(dists_uint8, threshold, 255, cv2.THRESH_BINARY, mask) 
    return mask

"""
player1: The move token player1 is using
player1Turn: whether it's player1's turn
move: The move token actually played
"""
def checkLegalMark(player1, player1Turn, move):
    if player1Turn:
        if player1 == move:
            return True
        else:
            return False
    else: #Player 2's turn
        if player1 == 'X' and move == 'O':
            return True
        elif player1 == 'O' and move == 'X':
            return True
        else:
            return False

"""
retrieves the board coordinates based on the projector pixel location
"""
def cellLocation(x, y):
    print "x, y:", x, y
    # then first column
    if x < squareSize:
        if y < squareSize:
            return (0,0)
        elif y < 2*squareSize:
            return (1,0)
        elif y < 3*squareSize:
            return (2,0)
    elif x < 2*squareSize:
        if y < squareSize:
            return (0,1)
        elif y < 2*squareSize:
            return (1,1)
        elif y < 3*squareSize:
            return (2,1)
    elif x < 3*squareSize:
        if y < squareSize:
            return (0,2)
        elif y < 2*squareSize:
            return (1,2)
        elif y < 3*squareSize:
            return (2,2)
    return False

"""
Fills a colored square in image
different colors depending on the player
"""
def colorCells(image, cells, base, col1, player1 = True):
    color = (255,255,0)
    if not player1:
        color = (0,255,0)
    for cell in cells:
        cv2.rectangle(image, (base+cell[1]*squareSize, col1+cell[0]*squareSize), (base+(1+cell[1])*squareSize, col1+(1+cell[0])*squareSize), color, -1)


winName = "Win"
cv2.namedWindow(winName, cv2.CV_WINDOW_AUTOSIZE)

# open the device.
capture = cv2.VideoCapture(0)
if capture:
    print 'Opened device number 0 - press Esc to stop capturing.'

# Bail if error.
if not capture:
    print 'Error opening video capture!'
    sys.exit(1)

# Fetch the first frame and bail if none.
ok, frame = capture.read()
if not ok or frame is None:
    print 'No frames in video'
    sys.exit(1)

w = frame.shape[1]
h = frame.shape[0]

squareSize = 80
base = 250 #for the x dimension
col1 = 250

#Initialize the board image
board = np.zeros((h, w, 3), dtype = np.uint8)
cv2.line(board, (base+squareSize, col1), (base+squareSize, col1+3*squareSize), (255,255,255), 10)
cv2.line(board, (base+2*squareSize, col1), (base+2*squareSize, col1+3*squareSize), (255,255,255), 10)
cv2.line(board, (base, col1+squareSize), (base+3*squareSize, col1+squareSize), (255,255,255), 10)
cv2.line(board, (base, col1+2*squareSize), (base+3*squareSize, col1+2*squareSize), (255,255,255), 10)


cv2.imshow(winName, board)
cv2.waitKey(1000)

################### TRAIN the KNN ############################
########################################################

# Load the data, converters convert the letter to a number
data= np.loadtxt('testdata.data', dtype= 'float32', delimiter = ',',
                    converters= {0: lambda ch: ord(ch)-ord('A')})

# split the data to two, 10000 each for train and test
train, test = np.vsplit(data,2)
train = data
test = data

# split trainData and testData to features and responses
responses, trainData = np.hsplit(train,[1])
labels, testData = np.hsplit(test,[1])

# Initiate the kNN, classify, measure accuracy.
knn = cv2.KNearest()
knn.train(trainData, responses)

############################ Construct the templates #######################
######################################################################

x_templates = []
o_templates = []

o_line_widths = [2, 3, 4]
o_temp_sizes = [35, 45, 55, 65, 75]

x_line_widths = [1, 2, 3, 4]
x_temp_sizes = [15, 25]


# Generate o_templates of dif sizes and line widths
for width in o_line_widths:
    for size in o_temp_sizes:
        o = np.zeros((size,size,3), dtype = np.uint8)
        cv2.circle(o, (size/2,size/2), (size/2)-5, (255,255,255), width)
        #x_templ = np.empty((size, size), 'uint8')
        #o_templ = np.empty((size, size), 'uint8')
        o_templ = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
        cv2.GaussianBlur(o_templ, (1,1), 1)
        o_templates.append(o_templ)

# Generate x_templates of diff sizes and line widths
for width in x_line_widths:
    for size in x_temp_sizes:
        x = np.zeros((size,size,3), dtype = np.uint8)
        cv2.line(x, (0,0), (size,size), (255,255,255), width)
        cv2.line(x, (0,size), (size, 0), (255,255,255), width)
        x_templ = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        cv2.GaussianBlur(x_templ, (1,1), 1)
        x_templates.append(x_templ)


temp = np.zeros((h, w, 3), dtype = np.uint8)
cv2.imshow(winName, temp)
cv2.waitKey(5000)

for i in range(10):
    ok, frame = capture.read()
    if not ok or frame is None:
        exit(1)

#Temporal average matrix
temporal_avg = frame #np.zeros((h, w, 3), dtype = np.float64)

cv2.imshow(winName, frame)
cv2.waitKey(3000)

# Create 9 dots roughly covering the intended game board
centers = [(base, col1),(base, col1+squareSize),(base, col1+2*squareSize),
            (base+80, col1),(base+80, col1+squareSize),(base+80,col1+2*squareSize),
            (base+160, col1),(base+160, col1+squareSize),(base+160, col1+2*squareSize)]

c1 = []
c2 = []

# Create the list of projector pixel locations to create equivalences with the
#   camera pixel locations to create the homography
for c in centers:
    c1.append(np.array(cvk2.a2ti(np.array(c)), dtype=np.float64))

# Project all the centers to get the webcame pixel locations for the homography
for center in centers:
    image = temp.copy()
    cv2.circle(image, center, 15, (255,255,255), -1)
    cv2.imshow(winName, image)
    cv2.waitKey(600)
    ok, frame = capture.read()
    if not ok or frame is None:
        exit(1)

    mask = subtractBackground(frame, temporal_avg)

    work = mask.copy()

    cv2.imshow(winName, mask)
    cv2.waitKey(200)

    # find the centroid of the dot and add it to c1
    cnt = cv2.findContours(work, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    c2.append(np.array(cvk2.a2ti(getDot(cnt)[0]), dtype=np.float64))


##############
#PART B
#############

homo = cv2.findHomography(np.array(c2), np.array(c1), method = cv.CV_RANSAC)
print homo
print

# Construct a transformation matrix for the image that achieves
# the desired rotation and translation
M = np.eye(3,3, dtype='float32')
M2 = np.matrix(M) * homo[0]

# parameters for the text displays
i = 100 #the x-pos on the screen
j = 100
t = 2

def clearText(img):
    img[j-40:j+100, 50:600] = (0,0,0)

def updateGuesses(letters, guesses, img):
    clearText(img)
    cv2.putText(img, "unused letters: "+' '.join(letters), (i, j), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (255,255,255), t, cv2.CV_AA)

    cv2.putText(img, "guessed letters: "+' '.join(guesses), (i, j+50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (255,255,255), t, cv2.CV_AA)

    return


player1Color = (255,200,0)
player2Color = (0,255,0)

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
guesses = ['Q']

#Initialize the images for all the different message boards
clearBoard = board.copy()
cv2.putText(clearBoard, "Guess a letter", (i, j-80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                player1Color, t, cv2.CV_AA)
updateGuesses(letters, guesses, clearBoard)

"""
for letter in letters:
    guesses.append(letter)
    del letters[0]
    updateGuesses(letters, guesses, clearBoard)

    cv2.imshow(winName, clearBoard)
    cv2.waitKey(2000)
    # opencv use x,y and numpy uses y,x
    #clearBoard[j+60:j+110, :] = (0,0,0)
    """


clearBoardGuessMade = board.copy()
cv2.putText(clearBoardGuessMade, "You guessed: ", (i, j), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                player2Color, t, cv2.CV_AA)

clearBoardUnknownLetter = board.copy()
cv2.putText(clearBoardUnknownLetter, "Unknown letter, erase and try again", (i, j),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0,0,255), t, cv2.CV_AA)

obstructedBoard = board.copy()
cv2.putText(obstructedBoard, "obstructed", (i, j), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0,0,255), t, cv2.CV_AA)


cv2.imshow(winName, clearBoard)
cv2.waitKey(500)

for i in range(10):
    ok, frame = capture.read()
    if not ok or frame is None:
        exit(1)

#Temporal average matrix
temporal_avg = frame[:]

# Bounds for hysteresis thresholding
x_ub = .82
x_lb = .77
o_ub = .49
o_lb = .59

# To display the image of the board that the webcam sees
board_view_mask = (slice(col1, col1+squareSize), 
                    slice(base+4*squareSize,base+5*squareSize))

# To display the thresholded image of the board
threshold_mask = (slice(col1, col1+squareSize), 
                    slice(base+6*squareSize,base+7*squareSize))

base += squareSize #for the x dimension
col1 += squareSize


cv2.imshow(winName, clearBoard)
cv2.waitKey(500)

gameplaying = True
player1Turn = True
player1 = ""
legalMove = True
gameBoard = {} #Represents tic tac toe board 

while gameplaying:
    ok, frame = capture.read()
    if not ok or frame is None:
        exit(1)

    # Warp the image to the destination in the temp image.
    dst = cv2.warpPerspective(frame, M2, (w, h))

    # The actual game board
    roi = dst[col1:col1+squareSize, base:base+squareSize]

    mask = subtractBackground(frame, temporal_avg, threshold=60)
    warped_mask = cv2.warpPerspective(mask, M2, (w, h))
    temp_mask_roi = warped_mask[col1:col1+squareSize, base:base+squareSize]
    # with value less than 100 go to 0, and all pixels with value
    # greater than or equal to 100 go to 255.
    mask_roi = cv2.threshold(temp_mask_roi, 100, 255, cv2.THRESH_BINARY)[1]

    eroded_mask = np.zeros(mask_roi.shape, dtype = np.uint8)
    kernel = np.ones((4,4),np.uint8)
    # erode any human made marks to detect obstructions
    cv2.erode(mask_roi, kernel, eroded_mask)

    display_mask = cv2.cvtColor(mask_roi, cv.CV_GRAY2RGB);

    # Board is obstructed so display obstruction message
    if np.sum(eroded_mask) > 25000:
        # copies the region of interest into the board for display
        obstructedBoard[board_view_mask] = roi[:,:]
        obstructedBoard[threshold_mask] = display_mask[:,:]
        cv2.imshow(winName, obstructedBoard)

    elif np.sum(mask_roi) > 15000:
        # If board isn't obstructed, search for templates
        # To search for templates check all x templates and all o templates.
        #   record the best result for each. Then using hysterisis thresholding
        #   check if the mark should be classified as x, o or neither

        ###################### Now finding Contours ##########################
        ######################################################################
        ## Citation: http://stackoverflow.com/questions/9413216/simple-digit-recognition-ocr-in-opencv-python?lq=1

        contours,hierarchy = cv2.findContours(mask_roi.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        [x2, y2, w2, h2] = [0]*4
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                [x2,y2,w2,h2] = cv2.boundingRect(cnt)

                #roi = thresh[y:y+h,x:x+w]
                #roismall = cv2.resize(roi,(10,10))
                #cv2.imshow('norm',im)
        
        if max_area > 50:
            cv2.rectangle(display_mask,(x2,y2),(x2+w2,y2+h2),(0,0,255),2)
            
        features = findattributes.getFeatures(mask_roi)
        ret, result, neighbours, dist = knn.find_nearest(
                        np.array([features], dtype='float32'), k=3)
        guess = unichr(int(result[0][0]) + ord('A'))
        print guess, "dist: ", dist, "features: ", features        


        #Now display the proper message
        if np.sum(dist) > 800:
            clearBoardUnknownLetter[board_view_mask] = roi[:,:]
            clearBoardUnknownLetter[threshold_mask] = display_mask[:,:]
            cv2.imshow(winName, clearBoardUnknownLetter)
        else:
            clearBoardGuessMade[board_view_mask] = roi[:,:]
            clearBoardGuessMade[threshold_mask] = display_mask[:,:]

            guesses[0] = guess
            #letters.remove(guess)
            clearText(clearBoardGuessMade)
           # """
            cv2.putText(clearBoardGuessMade, "You guessed: "+guess, (i, j), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                player2Color, t, cv2.CV_AA)
           # """

            cv2.imshow(winName, clearBoardGuessMade)

    else: #No marks were made, display turn message
#TODO: should we reset temporal avg if legal move was false
        legalMove = True
        clearBoard[board_view_mask] = roi[:,:]
        clearBoard[threshold_mask] = display_mask[:,:]
        cv2.imshow(winName, clearBoard)
        
    # Delay for 1ms and get a key
    k = cv2.waitKey(10)

    # Check for ESC hit:
    if k % 0x100 == 27:
        break


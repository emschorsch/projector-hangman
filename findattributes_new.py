import cv2
import cv
import numpy as np
import sys
import struct
import glob

# Tell python where to find cvk2 module before importing it.
sys.path.append('../cvk2')
import cvk2

def getImMask(filename):

    # Get letter mask
    image_rgb = cv2.imread(filename)

    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    w = image.shape[1]
    h = image.shape[0]
    mask = np.zeros((h, w), 'uint8')
    mask_resized = np.zeros((80,80),'uint8')
    cv2.threshold(image, 170, 255, cv2.THRESH_BINARY, mask)
    cv2.resize(mask,(80, 80), mask_resized)


    # For test letters - invert mask
    mask_resized = np.subtract(255, mask_resized)
    return mask_resized
    

def getFeatures(mask):
    w = mask.shape[1]
    h = mask.shape[0]

    # Get y,x indices of "on" pixels
    indices = np.array(np.where(mask==255))
    x = indices[1]                 # X measured from left
    y = np.subtract(h, indices[0]) # Y measured from bottom

    # Find smallest box around all "on" pixels
    xmin = min(x)
    ymin= min(y)
    xmax = max(x)
    ymax = max(y)
    x_center = (xmin+xmax)/2 # x center of bounding box
    y_center = (ymin+ymax)/2 # y center of bounding box (measured from bottom)
    a1 = x_center   # Horizontal position of box center 
    a2 = y_center   # Vertical position of box center (measured from bottom)
    a3 = xmax-xmin  # Width of box
    a4 = ymax-ymin  # Height of box

    # Display rectangle around letter
    frame = mask.copy()
    #cv2.rectangle(frame, (xmin, h-ymin), (xmax, h-ymax), 255)
    #cv2.imshow(win, frame)
    #cv2.waitKey()

    a5 = indices.shape[1] # Total number of "on" pixels

    # Get horiz pos. of pixels rel to cntr and divide by width of box
    h_pos = np.subtract(x,x_center)
    h_pos_scaled = np.divide(h_pos.astype(float),a3)
    a6 = np.average(h_pos_scaled)

    # Get vert pos. of pixels rel to cntr and divide by height of box
    v_pos = np.subtract(y,y_center)
    v_pos_scaled = np.divide(v_pos.astype(float),a4)
    a7 = np.average(v_pos_scaled)

    # Get mean squared horiz pixel distances
    h_pos_squared = np.square(h_pos_scaled) 
    a8 = np.average(h_pos_squared)

    # Get mean squared vert pixel distances
    v_pos_squared = np.square(v_pos_scaled)
    a9 = np.average(v_pos_squared)

    # Get mean product of horiz and vert distances
    hv_prod = np.multiply(h_pos_scaled,v_pos_scaled)
    a10 = np.average(hv_prod)

    # Get mean value of squared horiz distance times vertical dist
    # (correlation btwn horiz variance and vert pos)
    h2v_prod = np.multiply(h_pos_squared,v_pos_scaled)
    a11 = np.average(h2v_prod)

    # Get mean value of squared vert distance times horiz dist
    # (correlation btwn vert variance and horiz pos)
    v2h_prod = np.multiply(v_pos_squared,h_pos_scaled)
    a12 = np.average(v2h_prod)


    # Get image shifted one pixel to the right
    right_im = mask[:,0:w-1]
    col = np.zeros((h,1))
    right_im = np.hstack([col, right_im])
    # Crop image around letter
    right_im = right_im[(h-ymax):(h-ymin),xmin:xmax]
    #cv2.imshow(win,right_im)
    #cv2.waitKey()

    # Get difference image to find horizontal edges
    dif_im = np.subtract(mask[(h-ymax):(h-ymin),xmin:xmax],right_im)
    # Convert to edge mask (1 for edge, 0 otherwise)
    edge_im = (dif_im==255)
    edge_im = edge_im.astype(int)
    edgesbyrow = np.sum(edge_im, axis=1) # sum no. of edges in each row
    a13 = np.average(edgesbyrow) #average number of edges per row

    #cv2.imshow(win,dif_im)
    #cv2.waitKey()

    edge_indices = np.where(edge_im)
    # Vert position of edges measured from bottom of box (*need to make sure this is right)
    y_edge_pos = np.subtract(a4, edge_indices[0]) 
    a14 = sum(y_edge_pos)

    # Get image shifted one pixel up
    up_im = mask[1:h,:]
    row = np.zeros((1,w))
    up_im = np.vstack([up_im, row])

    # Crop image around letter
    up_im = up_im[(h-ymax):(h-ymin),xmin:xmax]
    up_im[ymax-ymin-1,:] = row[:,xmin:xmax]

    # Get difference image to find vertical edges
    dif_im = np.subtract(mask[(h-ymax):(h-ymin),xmin:xmax], up_im)
    #cv2.imshow(win,dif_im)
    #cv2.waitKey()

    edge_im = (dif_im==255)
    edge_im = edge_im.astype(int)
    edgesbycol = np.sum(edge_im, axis=0) # no. of edges in each row
    a15 = np.average(edgesbycol)

    edge_indices = np.where(edge_im)
    # Horiz position of edges measured from left of box 
    x_edge_pos = np.subtract(a3, edge_indices[1])
    a16 = sum(x_edge_pos)

    attributes = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16]
    #print attributes


    # Scale attributes to integers between 0 and 15
    # Newvalue = (((OldValue-OldMin) * NewRange) / OldRange) + NewMin
    oldmin1 = 1
    oldmax1 = 80
    oldrange1 = oldmax1 - oldmin1

    # Scale attributes with range 0:80 (for 80x80 pixel image)
    a1 = int(a1*15/80)
    a2 = int(a2*15/80)
    a3 = int(a3*15/80)
    a4 = int(a4*15/80)

    # Scale attribute with range 0:6400
    a5 = int(a5*15/6400)

    # Scale attributes with range -1:1
    a6 = int(((a6+1)*15)/2)
    a7 = int(((a7+1)*15)/2)
    a8 = int(((a8+1)*15)/2)
    a9 = int(((a9+1)*15)/2)
    a10 = int(((a10+1)*15)/2)
    a11 = int(((a11+1)*15)/2)
    a12 = int(((a12+1)*15)/2)

    # Scale attributes with range 0:40
    a13 = int(a13*15/40)
    a15 = int(a15*15/40)

    # Scale attributes with range 0:1600
    a14 = int(a14*15/3600)
    a16 = int(a16*15/3600)



    attributes = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16]
    print attributes
    return attributes

def main():
    alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    A_filenames = glob.glob("letters/A/*")
    #filename = "test_l.png"
    #mask = getImMask(filename)
    #features = getFeatures(mask)
    #print features

    f = open('testdata.data', 'w')
    

    for letter in alphabet:
        filenames = glob.glob("letters/" + letter + '/*')
        for filename in filenames:
            mask = getImMask(filename)
            features = getFeatures(mask)
            #print features
            f.write(letter + ',' + ','.join(map(str, features)) + "\n")

        
    
main()
    





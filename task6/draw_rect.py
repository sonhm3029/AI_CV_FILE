import cv2 as cv



""" Syntax: cv2.rectangle(image, start_point, end_point, color, thickness) """

# Parameters:
# image: It is the image on which rectangle is to be drawn.
# start_point: It is the starting coordinates of rectangle. The coordinates are represented as tuples of two values i.e. (X coordinate value, Y coordinate value).
# end_point: It is the ending coordinates of rectangle. The coordinates are represented as tuples of two values i.e. (X coordinate value, Y coordinate value).
# color: It is the color of border line of rectangle to be drawn. For BGR, we pass a tuple. eg: (255, 0, 0) for blue color.
# thickness: It is the thickness of the rectangle border line in px. Thickness of -1 px will fill the rectangle shape by the specified color.

# Return Value: It returns an image.
"""             Example                     """
image = cv.imread('C:\image.jpg')
start_point = (5,5)
end_point = (200, 200)

color = (255,0,0)

thickness = 2

image = cv.rectangle(image, start_point, end_point, color, thickness)

x, y, z = image.shape
print(x, y)

cv.imshow('Imgae', image)

cv.waitKey(0)
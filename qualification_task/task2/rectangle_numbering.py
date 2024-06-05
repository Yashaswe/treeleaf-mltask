import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('rectangles.png')

# Edge detection
edges = cv2.Canny(image, 50, 150)
cv2.imshow('Edge', edges)
# Find contours in the edge-detected image with hierarchical retrieval
# CHAIN_APPROX_SIMPLE: Removes all redundant points and only stores the edge points
# RETR_CCOMP: For extracting both internal and external contours, and organizing them into a two-level hierarchy
contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

contour_image = image.copy()

inner_contour_lengths = []

for i, contour in enumerate(contours):
    # Get the parent index from the hierarchy
    parent_idx = hierarchy[0][i][3]
    
    # Only work on the Inner contour
    if parent_idx != -1:
        
        # Find the arcLength or the length of the line
        perimeter = cv2.arcLength(contour, True)
        inner_contour_lengths.append((perimeter, i))

inner_contour_lengths.sort()

# Assign numbers to the lines based on their lengths
line_numbers = {}
for i, length_index in enumerate(inner_contour_lengths):
    line_index = length_index[1] 
    line_number = i + 1 
    line_numbers[line_index] = line_number


# Draw and label the lines for the four contours with lowest lengths
for length, index in inner_contour_lengths[:4]:  # Only the first four contours
    contour = contours[index]
    
    number = line_numbers[index]
    cv2.putText(contour_image, str(number), tuple(contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
cv2.imwrite("rectangles_ordered.png", contour_image)
cv2.imshow("Rectangles Ordered", contour_image)
cv2.waitKey(0) 
cv2.destroyAllWindows() 

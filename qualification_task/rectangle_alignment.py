import cv2
import numpy as np

def order_points(pts):
    # Initialize a list of coordinates that will be ordered
    # such that the first entry is the top-left, the second is the top-right,
    # the third is the bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Now, compute the difference between the points
    # The top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Return the ordered coordinates
    return rect

def align_rectangle_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny edge detection with adjusted parameters
    edges = cv2.Canny(blur, 50, 150)

    # Find contours of the rectangles
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out rectangles with line inside and approximate to 4 vertices
    rectangles = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            rectangles.append(approx)

    # Sort rectangles by area in descending order
    rectangles = sorted(rectangles, key=cv2.contourArea, reverse=True)

    # Process each rectangle
    for i, rectangle in enumerate(rectangles[:4]):  # Consider the largest four rectangles
        rect_points = rectangle.reshape(4, 2).astype(np.float32)
        rect_points = order_points(rect_points)  # Ensure the points are in the correct order

        # Calculate width and height of the detected rectangle
        width_a = np.linalg.norm(rect_points[0] - rect_points[1])
        width_b = np.linalg.norm(rect_points[2] - rect_points[3])
        max_width = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(rect_points[0] - rect_points[3])
        height_b = np.linalg.norm(rect_points[1] - rect_points[2])
        max_height = max(int(height_a), int(height_b))

        # Define the target points for perspective transformation
        target_points = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        # Calculate the perspective transformation matrix
        M = cv2.getPerspectiveTransform(rect_points, target_points)

        # Apply the perspective transformation to the rectangle to align it
        aligned_rectangle = cv2.warpPerspective(image, M, (max_width, max_height))
        
        # Save the aligned rectangle image
        cv2.imwrite(f"task2_aligned_rectangle_{i+1}.png", aligned_rectangle)

        # Optionally display the aligned rectangle
        cv2.imshow(f"Aligned Rectangle {i+1}", aligned_rectangle)

    # Draw rectangles on the original image
    for rectangle in rectangles[:4]:  # Draw the largest four rectangles
        cv2.drawContours(image, [rectangle], -1, (0, 0, 255), 2)

    cv2.imwrite(f"rectangles_ordered.png", image)
    # Display the original image with rectangles drawn on it
    cv2.imshow("Original Image with Rectangles", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Read the image
image = cv2.imread("rectangles.png")

# Align and display rectangles on the original image
align_rectangle_image(image)

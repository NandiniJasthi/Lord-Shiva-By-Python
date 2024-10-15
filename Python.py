import cv2
import numpy as np
from matplotlib import pyplot as plt
import requests
from io import BytesIO
from PIL import Image

# Define the function to find the closest point
def find_closest(p, positions):
    if len(positions) > 0:
        nodes = np.array(positions)
        distances = np.sum((nodes - p) ** 2, axis=1)
        i_min = np.argmin(distances)
        return positions[i_min]
    return None

# Define the function to apply adaptive thresholding for outlining the image
def outline(im):
    blurred = cv2.GaussianBlur(im, (7, 7), 0)
    th3 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                thresholdType=cv2.THRESH_BINARY, blockSize=9, C=2)
    return th3

# Download the image from the URL
image_url = 'https://i.postimg.cc/vTZd9LSR/71-WANKqe-Ea-L-AC-UF1000-1000-QL80.jpg'
response = requests.get(image_url)
img = Image.open(BytesIO(response.content)).convert('L')  # Convert to grayscale

# Convert PIL image to OpenCV format (numpy array)
im = np.array(img)

# Process the image to generate the thresholded outline
th3 = outline(im)

# Display the threshold image
plt.imshow(th3, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()

# Get the width and height of the image
WIDTH = im.shape[1]
HEIGHT = im.shape[0]
print("Image Width:", WIDTH, "Image Height:", HEIGHT)

# Calculate cutoff length
CUTOFF_LEN = ((WIDTH + HEIGHT) / 2) / 60

# Get the positions of black pixels (where the threshold is 0)
iH, iW = np.where(th3 == 0)
iW = iW - WIDTH / 2  # Center horizontally
iH = -1 * (iH - HEIGHT / 2)  # Center vertically and invert to match the plot

# Create a list of positions from the black pixel coordinates
positions = [list(iwh) for iwh in zip(iW, iH)]

# Initialize drawing simulation using matplotlib
fig, ax = plt.subplots()
ax.set_xlim(-WIDTH / 2, WIDTH / 2)
ax.set_ylim(-HEIGHT / 2, HEIGHT / 2)
ax.set_aspect('equal')
ax.set_facecolor('black')  # Set background color to match the drawing environment

# Start the drawing simulation
p = positions[0]  # Start from the first black pixel
ax.plot(p[0], p[1], 'ro')  # Starting point in red
plt.pause(0.1)

try:
    while p and len(positions) > 0:
        p = find_closest(p, positions)
        if p:
            ax.plot(p[0], p[1], 'bo', markersize=1)  # Plot the next position in blue
            plt.pause(0.001)  # Small delay to simulate drawing
            positions.remove(p)  # Remove the visited position
        else:
            break
except KeyboardInterrupt:
    print("Drawing interrupted by user")

# Show the final plot
plt.show()

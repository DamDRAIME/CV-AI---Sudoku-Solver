import numpy as np
import cv2

grid_path = "C:/Users/Admired AI Men/Documents/Projects/Sudoku Solver/Sudoku grids/Grid - 05.png"

org_grid = cv2.imread(grid_path, 0)

border_grid = cv2.copyMakeBorder(
    org_grid, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, 255
)

resized_grid = cv2.resize(border_grid, (600, 600), cv2.INTER_LINEAR)

processed_grid = cv2.GaussianBlur(resized_grid, (5, 5), 0)
processed_grid = cv2.adaptiveThreshold(
    resized_grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)
processed_grid = cv2.GaussianBlur(processed_grid, (13, 13), 0)
canny_grid = cv2.Canny(processed_grid, 255, 150, 7)

dilation = cv2.dilate(
    canny_grid, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1
)

contours, hierarchy = cv2.findContours(
    dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

resized_grid_color = cv2.cvtColor(resized_grid, cv2.COLOR_GRAY2BGR)

if len(contours) > 0:
    best_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(best_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    best_contour_grid = cv2.drawContours(resized_grid_color, [box], 0, (0, 255, 0), 2)


cv2.imshow("canny", canny_grid)
cv2.imshow("result", best_contour_grid)

cv2.waitKey(0)
cv2.destroyAllWindows()

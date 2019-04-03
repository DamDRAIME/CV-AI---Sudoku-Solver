import numpy as np
import cv2
import os

grid_folder = "C:/Users/Admired AI Men/Documents/Projects/Sudoku Solver/Sudoku grids/"
grid_path = grid_folder + os.listdir(grid_folder)[19]


org_grid = cv2.imread(grid_path, 0)


border_grid = cv2.copyMakeBorder(org_grid, 5, 5, 5, 5, cv2.BORDER_CONSTANT, None, 255)

denoised_grid = cv2.fastNlMeansDenoising(
    org_grid, h=3, templateWindowSize=7, searchWindowSize=7
)

resized_grid = cv2.resize(denoised_grid, (600, 600), cv2.INTER_LINEAR)

processed_grid = cv2.GaussianBlur(resized_grid, (7, 7), 0)
processed_grid = cv2.adaptiveThreshold(
    resized_grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)
processed_grid = cv2.GaussianBlur(processed_grid, (13, 13), 0)
canny_grid = cv2.Canny(processed_grid, 255, 150, 7)

dilation = cv2.dilate(
    canny_grid, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1
)

opening = cv2.morphologyEx(canny_grid, cv2.MORPH_OPEN, (3, 3))

contours, hierarchy = cv2.findContours(
    dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

contours_area = [
    cv2.contourArea(contour)
    for contour in contours
    if 3000 <= cv2.contourArea(contour) <= 10000
]

resized_grid_color = cv2.cvtColor(resized_grid, cv2.COLOR_GRAY2BGR)

all_contours = cv2.drawContours(resized_grid_color.copy(), contours, -1, (0, 255, 0), 2)

if len(contours) > 0:
    best_contour = max(contours, key=cv2.contourArea)
    # box = cv2.boundingRect(best_contour)
    # best_contour_grid = cv2.rectangle(
    #     resized_grid_color.copy(), (box[0],box[1]), (box[2],box[3]), (0, 255, 0), 2
    # )
best_contour_area = cv2.contourArea(best_contour)

if len(contours_area) >= 10 or best_contour_area <= 90000:
    print("Assume full frame")
# best_contour_bis = cv2.approxPolyDP(best_contour, 5, True)

best_contour = cv2.drawContours(
    resized_grid_color.copy(), [best_contour], 0, (0, 255, 0), 2
)

# best_contour_bis = cv2.drawContours(
#     resized_grid_color.copy(), [best_contour_bis], 0, (0, 255, 0), 2
# )

cv2.imshow("canny", canny_grid)
cv2.imshow("all_contours", all_contours)
cv2.imshow("best_contour", best_contour)
# cv2.imshow("best_contour_bis", best_contour_bis)
cv2.imshow("resized_grid", resized_grid)

cv2.waitKey(0)
cv2.destroyAllWindows()

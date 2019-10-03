# CV-AI---Sudoku-Solver
End-to-end Sudoku solver in Python. From the image of a Sudoku grid, our goal will be to create a Sudoku solver that returns the solved grid to the user.

## Sudoku Solver
As mentioned, the input to our Sudoku solver will be an image of a Sudoku grid. Hence two obstacles have to be overcome before tackling the constraints problem (i.e. the Sudoku).

We indeed need to first parse the image so that the digits from all non-blank cells are extracted. Only once this pre-filled grid has been extracted, Constraint Logic Programming techniques can be applied to solve the Sudoku. 

Here is a high level overview of the pseudo code for the Sudoku solver
```python
def sudoku_solver(image):
  image_processed = preprocessing(image)
  corners = grid_detection(image_processed)
  extracted_grid = []
  For each cell in extract_cells(image_processed, corners):
    digit_pred = digit_recognition(cell)
    extracted_grid.append(digit_pred)
  solved_grid = clp(extracted_grid)
  return solved_grid
```
Of course, all of this relies on a data set.

Let's review the different steps involved:
### Step 00 - Gathering data
First and foremost, data need to be collected and labeled. By scraping the internet and manually taking pictures, 45 Sudoku grids were obtained. Some digital, other from newspapers.

| Digital Sudoku grid | "Newspaper" Sudoku Grid |
| --- | --- |
|![Digital Sudoku grid](/Sudoku%20grids/original/Grid-01.png?raw=true "Digital Sudoku grid")  |  ![Newspaper Sudoku grid](/Sudoku%20grids/original/Grid-15.jpg?raw=true "Newspaper Sudoku grid")|

In order to train a model to detect the grid in the image (i.e. its four corners), I manually located the four corners of the grid present in each image. Those (x, y) coordinates were saved in txt files. This covers the first data set.


| Formatted Sudoku grid | Corners coordinates |
| --- | --- |
|![](/Sudoku%20grids/formatted/Grid-12.jpg?raw=true "Formatted Sudoku grid")  |    TL;17;12<br/>TR;263;19<br/>BR;263;296<br/>BL;11;296 |

Similarly, to train a model to recognise the digit (or the absence of a digit) in a cell, each cell from each of the 45 grids had to be extracted and labeled. Then this second data set had to be balanced because it contained much more examples of blank cell than cell containing a digit. This could have influenced the model to only predict that a cell does not contain any digit in all cases. Here is an overview of this second data set:


| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | blank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|![](/digits/1/0_16.jpg?raw=true) ![](/digits/1/12_17.jpg?raw=true) ![](/digits/1/18_55.jpg?raw=true) ![](/digits/1/33_48.jpg?raw=true) |![](/digits/2/10_43.jpg?raw=true) ![](/digits/2/1_31.jpg?raw=true) ![](/digits/2/27_32.jpg?raw=true) ![](/digits/2/2_8.jpg?raw=true) |![](/digits/3/19_25.jpg?raw=true) ![](/digits/3/21_32.jpg?raw=true) ![](/digits/3/28_46.jpg?raw=true) ![](/digits/3/43_25.jpg?raw=true) |![](/digits/4/13_65.jpg?raw=true) ![](/digits/4/14_27.jpg?raw=true) ![](/digits/4/18_0.jpg?raw=true) ![](/digits/4/2_11.jpg?raw=true) |![](/digits/5/16_66.jpg?raw=true) ![](/digits/5/17_3.jpg?raw=true) ![](/digits/5/18_40.jpg?raw=true) ![](/digits/5/40_77.jpg?raw=true) |![](/digits/6/15_7.jpg?raw=true) ![](/digits/6/19_12.jpg?raw=true) ![](/digits/6/44_8.jpg?raw=true) ![](/digits/6/4_60.jpg?raw=true) |![](/digits/7/6_18.jpg?raw=true) ![](/digits/7/8_42.jpg?raw=true) ![](/digits/7/43_18.jpg?raw=true) ![](/digits/7/3_50.jpg?raw=true) |![](/digits/8/0_50.jpg?raw=true) ![](/digits/8/23_30.jpg?raw=true) ![](/digits/8/31_11.jpg?raw=true) ![](/digits/8/38_62.jpg?raw=true) |![](/digits/9/14_13.jpg?raw=true) ![](/digits/9/2_17.jpg?raw=true) ![](/digits/9/9_53.jpg?raw=true) ![](/digits/9/4_12.jpg?raw=true) |![](/digits/0/14_8.jpg?raw=true) ![](/digits/0/21_70.jpg?raw=true) ![](/digits/0/29_3.jpg?raw=true) ![](/digits/0/41_0.jpg?raw=true) |
### Step 01 - Grid detection
The first data set was thus used to train a CNN model to detect a grid in an image. This grid can be fully defined by its four corners. Hence the CNN model had to predict 8 values (i.e. the x and y coordinates of each corner) by doing a regression task.
### Step 02 - Digit recognition
### Step 03 - Constraints Problems Solver
## Usage
## Requirements


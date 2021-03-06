# CV-AI---Sudoku-Solver
End-to-end Sudoku solver in Python. From the image of a Sudoku grid, our goal will be to create a Sudoku solver that returns the solved grid to the user.

## Sudoku Solver
As mentioned, the input to our Sudoku solver will be an image of a Sudoku grid. Hence two obstacles have to be overcome before tackling the constraint satisfaction problem (i.e. the Sudoku).

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
The first data set was thus used to train a CNN to detect a grid in an image. This grid can be fully defined by its four corners. Hence the CNN model had to predict 8 values (i.e. the x and y coordinates of each corner) by doing a regression task.

Since the input shape of CNN is determined beforehand, all the image had to be formatted. Images were converted to greyscaled image with shape (305, 305, 1). A border was added to the image has it helped detect the grid border for digital Sudoku grids that filled the frame. The data set containing 45 examples was also artificially augemented during training. The challenge was to build a data augementation function that would output new examples while making sure that each one of those contains the full grid. Indeed, it can be that after augmentation (i.e. a series of affine transformations) some corners have vanished/ are outside the frame. The coordinates will also have to be adjusted and ouputed by the new data augementation function.

The CNN structure that was used for this task is given below:
```Python
input_grid = Input(shape=input_dim)
x = Conv2D(32, (7, 7), activation='relu', padding='same')(input_grid)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(256, activation='tanh')(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(output_dim, activation='linear')(x)
```

Once the four corners of the grid are predicted we can unwarp the grid (which is usually the case for grids from newspaper), and extract the cells. Those cells will then be processed by the second CNN.

Since the data set was quite small in size, the detection of corners is not very precise. Hence, it is possible to display a window asking the user to validate the predicted coordinates of the corners or modify those.
### Step 02 - Digit recognition
This second CNN is used to recognise what a cell contains. The second data set was used to train this CNN. This data set was also augmented via affine transformations.

The CNN structure that was used for this task is given below:
```Python
input_digit = Input(shape=input_dim)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_digit)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(output_dim, activation='softmax')(x)
```

The average accuracy of our CNN trained for 12 epochs was 0.916 on the test set. Here is the confusion matrix on test set:

![](/CM-digit.JPG?raw=true)

As we can see our digit recognition model has the most difficulty with the digit 6 which is often mistaken for a 4. Note that it is possible to display a window asking the user to validate the predicted digits or modify those predictions.

### Step 03 - Constraint Satisfaction Problems Solver
Finally, to solve the Sudoku, I used the [python-constraint library](https://pypi.org/project/python-constraint/), which can easily be installed with the following command line:
```
pip install python-constraint
```

## Usage
To solve a Sudoku grid for which you have an image, you can call the `solve_sudoku()` function. Let me illustrate this through an example. Here is the grid I will be using for it:

<img src="/Sudoku%20grids/original/Grid-23.jpg" width="250" height="250">

I pass as the main argument of the `solve_sudoku()` function the path to my Sudoku grid.
```Python
my_sudoku_grid = r"path/to/my/grid.png"
my_grid_detection_model_path = r"path/to/grid_detection_model.h5"
my_digit_recognition_model_path = r"path/to/digit_recognition_model.h5"
solved_grid = solve_sudoku(image_path = my_sudoku_grid, grid_detection_model_path = my_grid_detection_model_path,
                          digit_recognition_model_path = my_digit_recognition_model_path)
```
As mentioned above, the grid detection model is not very precise due to the lack of training data. Hence, by default, a window will pop up asking the user to confirm that the detected corners are the good ones before going any further.

```
>>>Here are the corners detected by the Sudoku Solver:
```
<img src="/Capture-grid-detection.JPG" width="250" height="250">

As you can see, in this example, the detected corners are not perfectly located. The user can remedy to this by clicking on the corners of the grid. (Be careful about the order). If nothing needs to be modified, then the user can simply close this window and the rest of the program will execute. Since we had to modify the corners coordinates in our example, we got a confirmation message after we have manually clicked on the four corners:

```
>>>Thanks for your input. The following coordinates were saved:
[[ 52  43]
 [277  42]
 [280 274]
 [ 52 281]]
```

Now each cell will be extracted from the grid and the digit it contains (or absence of digit) will be predicted by the second CNN. Again here, we can ask the user intervention in case we are not confident with our model. If we do so, then such a message will appear:

```
>>>Here is the grid as detected by the Sudoku Solver:
[[0 8 9 0 0 0 1 2 0]
 [2 0 0 1 0 8 0 0 9]
 [0 0 0 0 0 0 0 0 0]
 [1 0 0 4 0 7 0 0 8]
 [0 5 8 0 0 0 4 7 0]
 [4 0 0 9 0 5 0 0 6]
 [0 0 0 0 0 0 0 0 0]
 [8 0 0 6 0 1 0 0 7]
 [0 2 6 0 0 0 5 8 0]]
Are there some corrections to be done? (y/n)
```
In our case, no modification needs to be done and thus we can input a `n` in the terminal. In case corrections would be necessary, press `y` and follow the instructions.

_Note that you can disable these user intervention messages for both the corners detection and the digits recognition by passing `false` to the arguments `allow_user_coord_correction`, and `allow_user_digit_correction`, respectively, in the main function._

Now that the image is successfully parsed into a grid, the next step is to solve the Sudoku.
```
>>>Here is the solved Sudoku grid:
[[6 8 9 5 7 4 1 2 3]
 [2 3 5 1 6 8 7 4 9]
 [7 4 1 2 9 3 8 6 5]
 [1 6 3 4 2 7 9 5 8]
 [9 5 8 3 1 6 4 7 2]
 [4 7 2 9 8 5 3 1 6]
 [5 1 7 8 3 2 6 9 4]
 [8 9 4 6 5 1 2 3 7]
 [3 2 6 7 4 9 5 8 1]]
```

## Requirements
- Python 3.x
- numpy
- [python-constraint](https://pypi.org/project/python-constraint/)
- cv2
- matplotlib
- tensorflow
- keras


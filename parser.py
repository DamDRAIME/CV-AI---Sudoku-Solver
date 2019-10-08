import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import solver
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Model


grids_folder = "C:/Users/Admired AI Men/Documents/Projects/Sudoku Solver/Sudoku grids/original"
grids_folder_result = "C:/Users/Admired AI Men/Documents/Projects/Sudoku Solver/Sudoku grids/formatted"
coords_folder = "C:/Users/Admired AI Men/Documents/Projects/Sudoku Solver/Sudoku grids/coordinates"
digits_folder = "C:/Users/Admired AI Men/Documents/Projects/Sudoku Solver/digits"


def format_grids(original_grids_folder: str, formatted_grids_folder: str):
    """
    Function that will format all grids found in the folder original_grids_folder and save these formatted grids in
    the folder formatted_grids_folder
    :param original_grids_folder: Path to the folder containing the original grids to be formatted
    :param formatted_grids_folder: Path to the folder containing the resulting formatted grids
    :return: None
    """

    for grid_file in os.listdir(original_grids_folder):
        grid_path = os.path.join(original_grids_folder, grid_file)

        org_grid = cv2.imread(grid_path, 0)

        resized_grid = cv2.resize(org_grid, (300, 300), cv2.INTER_LINEAR)

        border_grid = cv2.copyMakeBorder(resized_grid, 5, 5, 5, 5, cv2.BORDER_CONSTANT, None, 255)

        cv2.imwrite(os.path.join(formatted_grids_folder, grid_file), border_grid)


def format_grid(image: np.ndarray):
    """
    Function to format an image/grid. Image will be standardized, resized, and a border will be added.
    :param image: image that needs to be formatted
    :return: np.ndarray - formatted image/grid
    """

    if image.shape[-1] == 1:
        image = image.squeeze(-1)
    image = cv2.resize(image, (300, 300), cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, None, 255)

    #  Standardization
    image_std = image.copy()
    image_std -= np.mean(image_std)
    image_std /= np.std(image_std)

    return np.expand_dims(image, axis=-1), np.expand_dims(image_std, axis=-1)


def load_corners_coordinates(coords_folder: str):
    """
    Function to load corners' coordinates from text files and generate a np.ndarray of shape (N, 4, 2),
    where N is the number of instances, 4 refers to the four corners (TL=TopLeft, TR=TopRight, BR=BottomRight,
    BL=BottomLeft), and 2 to the x and y coordinates. The order of the returned corners is given in the returned
    corners parameter.
    :param coords_folder: Path to the folder containing the text files with the coordinates
    :return: np.ndarray - Array (N, 4, 2) with coordinates; list - Corners order
    """

    first_pass = True
    coord_corners = []

    for coord_file in os.listdir(coords_folder):
        coord_path = os.path.join(coords_folder, coord_file)

        if first_pass:
            corners = list(np.loadtxt(coord_path, delimiter=';', dtype='S11', usecols=0))
            first_pass = False

        #  Make sure that all files have the same structure by comparing the first file structure with the others
        assert corners == list(np.loadtxt(coord_path, delimiter=';', dtype='S11', usecols=0)), \
            'Missing corner(s) or different order in file: {}'.format(coord_path)

        coord_corners.append(np.loadtxt(coord_path, delimiter=';', dtype='i4', usecols=(1, 2)))

    return np.array(coord_corners), corners


def load_grids(grids_dir: str):
    """
    Function to load grid images in a np.ndarray of shape (#Grids, Height, Width, 1). Images are loaded with grayscale.
    :param grids_dir: Path to the folder containing the grid images to be loaded
    :return: np.ndarray - Array (#Grids, Height, Width, 1)
    """

    x = []

    for grid_file in os.listdir(grids_dir):
        grid_path = os.path.join(grids_dir, grid_file)
        x.append(cv2.imread(grid_path, 0))

    x = np.array(x)
    x = np.expand_dims(x, axis=-1)

    return x.astype('float')


def load_digits(digits_dir: str, verbose: int=1):
    """
    Function to load digits images in a np.ndarray along with their respective labels.
    :param digits_dir: Main directory containing folders with the different digits. The assumed structure
    of digits_dir is the following:
        \digits_dir
            \0
                \img0.png
                \img1.png
            \1
                \img2.png
            \2
                \img3.png
                \img4.png
                \img5.png
            .
            .
            .
            \9
                \imgN.png
    Note that the sub-folders' names are important. Folder '0' should contain the images of empty cells. Images' names
    can be chosen freely. All images should have the same dimensions.
    :param verbose: 0 or 1
    :return: np.ndarray - Array with the digits images; np.ndarray - Array with the corresponding labels
    """

    if verbose >= 1:
        frequency_dict = {}
        nbr_instances = 0
        for folder, _, files in os.walk(digits_dir):
            if files:
                nbr_files = len(files)
                nbr_instances += nbr_files
                frequency_dict[folder] = nbr_files
                print('{} files loaded from folder {}'.format(nbr_files, folder))
    else:
        nbr_instances = sum([len(files) for _, _, files in os.walk(digits_dir)])

    img_shape = cv2.imread(os.path.join(digits_dir, '1', os.listdir(os.path.join(digits_dir, '1'))[0])).shape
    digits = np.zeros((nbr_instances, img_shape[0], img_shape[1]))
    labels = np.zeros((nbr_instances, 10))
    index = 0
    for digit in os.listdir(digits_dir):
        for instance in os.listdir(os.path.join(digits_dir, digit)):
            digits[index] = cv2.imread(os.path.join(digits_dir, digit, instance), 0)
            labels[index, int(digit)] = 1
            index += 1
    digits = digits.reshape((digits.shape[0], digits.shape[1], digits.shape[2], 1))

    return digits, labels


def unwarp_img(image: np.ndarray, landmark_coord: np.ndarray, new_dim: tuple=(306, 306)):
    """
    Function to unwarp the perspective of an image given by the four coordinates.
    In our application; unwarps the sudoku grid based on its four detected corners.
    :param image: Image for which we need to unwarp the perspective
    :param landmark_coord: Four (x,y) coordinates defining the current perspective on the image
    :param new_dim: Dimensions for the returned image
    :return: np.ndarray - Unwarped image
    """

    dst = np.array([
        [0, 0],
        [new_dim[0], 0],
        [new_dim[0], new_dim[1]],
        [0, new_dim[1]]], dtype="float32")
    ldmk = landmark_coord.astype(np.float32)

    # Compute the perspective transform matrix. Note that the function requires
    # the two arguments to be np.ndarrays with dtype=float32
    transform_matrix = cv2.getPerspectiveTransform(ldmk, dst)
    unwarped = cv2.warpPerspective(image, transform_matrix, new_dim)

    return unwarped


def _onclick(event):
    """
    Internal function to store the coordinates of the mouse when clicked.
    :param event: a mouse click
    :return: None
    """

    x, y = event.xdata, event.ydata

    #  The first click should remove the four corners predicted by the CNN
    if not user_coords:
        for scatter in scatters:
            scatter.remove()

    #  user_coords stores the coordinates of the mouse when clicked.
    user_coords.append([x, y])

    #  Only the first 4 clicks are taken into account
    if len(user_coords) == 4:
        plt.close('all')

    ax.scatter(x, y)
    plt.show()


def visualize_img_ldmk(image: np.ndarray, landmark_coord: np.ndarray, title: str='Sudoku and corners',
                       user_input: bool=False):
    """
    Function to visualize an grayscaled image along with some landmarks.
    In our application; visualizes the sudoku grid with its four corners.
    :param image: Image to be visualized
    :param landmark_coord: Landmarks' coordinates (x,y) to be displayed on top of the image
    :param title: Title given to the plot
    :param user_input: Record the coordinates of the mouse when clicked by the user
    :return: None
    """

    fig = plt.figure(figsize=(5, 5))
    if user_input:
        plt.title('Close the window if no corrections need to be made, \n'
                  'or click on the four corners starting from the one on the top left, \n'
                  'and then clock wise to the bottom left corner', fontsize=6, ha='center')
        fig.canvas.mpl_connect('button_press_event', _onclick)
    global ax
    ax = fig.add_subplot(1, 1, 1)
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = np.squeeze(image, axis=-1)
    ax.imshow(image, cmap='gray')
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    global scatters
    scatters = []
    for (x, y) in landmark_coord:
        scatters.append(ax.scatter(x, y))
    plt.suptitle(title)
    plt.axis('off')
    plt.show()


def extract_cells_from_grid(grid: np.ndarray):
    """
    Function to extract, one by one, the 81 cells from a unwarped sudoku grid.
    :param grid: Grid from which the digits need to be extracted. Assumes that the grid is unwarped.
    :return: np.ndarray - cells/digits (yield)
    """

    h, w = grid.shape

    # Sanity check
    assert h % 9 == 0 and w % 9 == 0, 'The height and width of the image should be divisible by 9.'

    # A window is convoluted across the grid. As such the stride should be the length of a grid's cell so that
    # extracted patches do not overlap each other. The window is configured to be
    # the size of a cell (i.e. grid's width divided by 9 and grid's height divided by 9)
    stride = w//9
    window_size = (h//9, w//9)
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Yield the current window/cell
            yield grid[y:y + window_size[1], x:x + window_size[0]]


def extract_and_save_cells_from_grids(grids_folder: str, coords_folder: str, digits_folder: str):
    """
    Function
    :param grids_folder: Folder containing the different grids to be processed. Assume that the grids are formatted.
    :param coords_folder: Folder containing the text files with the corners' coordinates for each grid
    :param digits_folder: Folder where the output should be saved
    :return: None
    """

    # To unwarp a grid we need the coordinates of its four corners
    coords, _ = load_corners_coordinates(coords_folder)

    for grid_idx, (grid_file, coord) in enumerate(zip(os.listdir(grids_folder), coords)):
        grid_path = os.path.join(grids_folder, grid_file)
        grid = cv2.imread(grid_path, 0)

        # extract_digits_from_grid() expects a unwarped grid
        grid_unwarped = unwarp_img(grid, coord)

        for digit_idx, digit in enumerate(extract_cells_from_grid(grid_unwarped)):

            # Save the extracted cell/patch
            cv2.imwrite(os.path.join(digits_folder, str(grid_idx+36) + '_' + str(digit_idx) + '.jpg'), digit)


def user_coord_correction(image: np.ndarray, corner_coords: np.ndarray):
    """
    Function used to ask a user input on the predicted corners' coordinates.
    :param image: image of the formatted Sudoku grid that was used for the prediction
    :param corner_coords: coordinates of the four corners that have been predicted by the CNN model
    :return: np.ndarray - corner_coords if the user didn't wish to edit them or the new coordinates provided by the user
    """

    global user_coords
    user_coords = []

    print('Here are the corners detected by the Sudoku Solver:')
    visualize_img_ldmk(image, corner_coords, 'Detected corners', True)

    if len(user_coords) == 4:
        user_coords = np.array([[int(round(x, 0)), int(round(y, 0))] for x, y in user_coords])
        print('Thanks for your input. The following coordinates were saved:')
        print(user_coords)
        return user_coords
    else:
        print('You did not wish to make any corrections to the predicted coordinates of the four corners made by '
              'the Sudoku Solver.')
        return corner_coords


def user_digit_correction(grid: list):
    """
    Function used to ask a user input on the predicted digits of the grid. Zeros symbolize blank cells.
    :param grid: list of all the digits (and blank cells) that were predicted by the CNN model
    :return: list - the same grid or the grid altered by user inputs
    """

    from ast import literal_eval
    print('Here is the grid as detected by the Sudoku Solver:')
    while True:
        print(grid)
        y_n = input('Are there some corrections to be done? (y/n) ')
        if y_n.lower() == 'n':
            break
        elif y_n.lower() == 'y':
            corrections_str = input('Provide your corrections as a python dictionary with '
                                    'key = (y_idx: int, x_idx: int), and value = correction: int. Separate each '
                                    'key/value pair with a semicolon (;). \n')
            corrections_str = corrections_str[1:-1].replace(" ", "")
            try:
                if ';' not in corrections_str:
                    corrections_str += ';'
                for key_value_pair in corrections_str.split(';'):
                    if len(key_value_pair) == 0:
                        break
                    key, value = key_value_pair.split(':')
                    y, x = literal_eval(key)
                    correction = int(value)

                    assert [type(y), type(x), type(correction)] == [int, int, int], \
                            'Elements from the key tuple should be integers. The value should be an integer'
                    assert y in range(0, 9) and x in range(0, 9), 'y_idx and x_idx should be between [0;8] as it ' \
                                                                  'designates the row and the column indices' \
                                                                  ', respectively.'
                    assert correction in range(0, 10), 'The value of a dictionary key corresponds to the new ' \
                                                       'value that should have the cell of the grid designated ' \
                                                       'by the key tuple. As such, this integer can only be ' \
                                                       'between [0;9].'
                    grid[y][x] = correction
                print('Your corrections have been made. Please verify.')
            except AssertionError as error:
                print(error)
            except:
                print('Your corrections were not provided as a python dictionary with '
                      'key = (y_idx: int, x_idx: int): tuple, and value = correction: int.')
                print('Example:')
                print('> grid = {}'.format([[1, 2, 0], [2, 5, 0], [9, 3, 0]]))
                print('> corrections = {(1, 2): 4; (0, 0): 7}')
                print('> corrected_grid = {}'.format([[7, 2, 0], [2, 5, 4], [9, 3, 0]]))

        else:
            print('Your command was not understood.')

    return grid


def parse_sudoku_grid(image_path: str, grid_detection_model_path: str, digit_recognition_model_path: str,
                      allow_user_digit_correction: bool=True, allow_user_coord_correction: bool=True):
    """
    Function to parse a Sudoku grid. Starting from an image of the Sudoku grid, the parser will first format it, then
    detect the grid to unwarp it, and finally detect and recognize the digits.
    :param image_path: path to the original image of the grid
    :param grid_detection_model_path: path to the .h5 model for grid/corners detection
    :param digit_recognition_model_path: path to the .h5 model for digit recognition
    :param allow_user_digit_correction: True = allow the user to verify and alter the predicted digits; False = contrary
    :param allow_user_coord_correction: True = allow the user to verify and alter the predicted corners; False = contrary
    :return: np.ndarray - Array (9, 9) predicted digits
    """

    img = cv2.imread(image_path, 0).astype('float')

    # Part 1: preformatting, detecting the grid, and unwarping it
    formatted_img, formatted_std_img = format_grid(img)
    grid_model = load_model(grid_detection_model_path)
    corners_coord = np.rint(grid_model.predict(np.array([formatted_std_img])))
    corners_coord = corners_coord.reshape((4, 2))

    if allow_user_coord_correction:
        corners_coord = user_coord_correction(formatted_img, corners_coord)

    unwarped_img = unwarp_img(formatted_std_img, corners_coord)

    # Part 2: extracting cells, and recognizing digits
    digit_model = load_model(digit_recognition_model_path)
    grid = []

    for cell in extract_cells_from_grid(unwarped_img):
        cell = np.array([np.expand_dims(cell, axis=-1)])
        grid.append(np.argmax(digit_model.predict(cell)))

    grid = np.array(grid).reshape((9, 9))

    if allow_user_digit_correction:
        grid = user_digit_correction(grid)

    return grid


def solve_sudoku(image_path: str, grid_detection_model_path: str, digit_recognition_model_path: str,
                 allow_user_digit_correction: bool=True, allow_user_coord_correction: bool=True):
    """
    Function to solve a Sudoku grid. Starting from an image of the Sudoku grid, the parser will first format it, then
    detect the grid to unwarp it, and finally detect and recognize the digits. Then the solver will solve the grid.
    :param image_path: path to the original image of the grid
    :param grid_detection_model_path: path to the .h5 model for grid/corners detection
    :param digit_recognition_model_path: path to the .h5 model for digit recognition
    :param allow_user_digit_correction: True = allow the user to verify and alter the predicted digits; False = contrary
    :param allow_user_coord_correction: True = allow the user to verify and alter the predicted corners; False = contrary
    :return: np.ndarray - Array (9, 9) predicted digits
    """
    #  Parsing the image to get the grid
    grid = parse_sudoku_grid(image_path, grid_detection_model_path, digit_recognition_model_path,
                             allow_user_digit_correction, allow_user_coord_correction)

    #  Solving the grid
    solved_grid = solver.solve_grid(grid)

    print('Here is the solved Sudoku grid:')
    print(solved_grid)
    return solved_grid


def digit_recognition(X: np.ndarray, y: np.ndarray):
    """
    Function used to train a CNN model to recognize digits.
    :param X: array of shape (N, X, Y, 1) containing the grids. Can be obtained with load_grids()
    :param y: array of shape (N, 4, 2) containing the ground-truth coordinates for the corners. Can be obtained
    with load_corners_coordinates()
    :return: np.ndarray - Array with the ground-truth labels for the test set;
    np.ndarray - Array with the predicted labels for the test set; model
    """

    datagen_args = dict(
        samplewise_center=True,
        samplewise_std_normalization=True,
        rotation_range=5,
        width_shift_range=3,
        height_shift_range=3,
        horizontal_flip=False,
        vertical_flip=False)
    datagenerator = ImageDataGenerator(**datagen_args)

    nbr_epochs = 12

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y)

    x_test -= np.mean(x_test)
    x_test /= np.std(x_test)

    input_dim = x_train[0].shape
    input_digit = Input(shape=input_dim)
    output_dim = y_train.shape[1]
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_digit)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(output_dim, activation='softmax')(x)
    model = Model(input_digit, x)
    model.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(datagenerator.flow(x_train, y_train, batch_size=32),
                        steps_per_epoch=len(x_train) / 32,
                        epochs=nbr_epochs,
                        shuffle=True,
                        validation_data=(x_test, y_test),
                        verbose=1)
    y_pred = model.predict(x_test)

    return y_test, y_pred, model


class ImageDataGenerator_landmarks(object):
    def __init__(self,
                 datagen_args,
                 preprocessing_function=None):
        """
        ImageDataGenerator for Keras that can take as input an image and a set of landmarks coordinates. Images are
        transformed through affine transformations and returned. The coordinates are also adjusted accordingly.
        If a transformed image results in the vanishing of at least one landmark, this transformed image is discarded
        from the output.
        :param datagen_args: Keras's ImageDataGenerator arguments
        :param preprocessing_function: The function that will be applied to each input.
        """

        self.datagen = ImageDataGenerator(**datagen_args)
        self.mask_channel = None
        self.nbr_landmarks = None
        self.batch_size = None

        if preprocessing_function is None:
            self.preprocessing_function = False
        elif preprocessing_function == 'standardization':
            self.preprocessing_function = lambda x, y: (self.standardization(x), y)
        else:
            self.preprocessing_function = preprocessing_function

    def standardization(self, x):
        x -= np.mean(x)
        x /= np.std(x)
        return x

    def flow(self, img_with_mask, batch_size):
        self.batch_size = batch_size
        generator = self.datagen.flow(img_with_mask, batch_size=self.batch_size)

        while True:
            N = 0
            x_bs, y_bs = [], []
            while N < self.batch_size:
                img_with_mask_batch = generator.next()
                x_batch, y_batch = self._keep_only_valid_instances(img_with_mask_batch)
                if len(x_batch) == 0:
                    continue
                if self.preprocessing_function is not False:
                    x_batch, y_batch = self.preprocessing_function(x_batch, y_batch)
                x_bs.append(x_batch)
                y_bs.append(y_batch)
                N += x_batch.shape[0]
            x_batch, y_batch = np.vstack(x_bs), np.vstack(y_bs)
            yield [x_batch, y_batch]

    def _keep_only_valid_instances(self, img_with_mask):
        x_train, y_train = [], []

        for img_with_mask_instance in img_with_mask:
            x_img = img_with_mask_instance[:, :, :self.mask_channel]
            y_mask = img_with_mask_instance[:, :, self.mask_channel]
            y_coord = self._find_coord_from_mask(y_mask)

            if y_coord is None:  # if some landmarks disappear, do not use the translated image
                continue

            x_train.append(x_img)
            y_train.append(y_coord)

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        return x_train, y_train

    def _find_coord_from_mask(self, y_mask):
        y_coord = []

        for i in range(1, self.nbr_landmarks + 1):
            y_mask = np.rint(y_mask)  # Needed when using rotation and shearing techniques as the integers can
                                    # be transformed to floats
            ix, iy = np.where(y_mask == i)

            if len(ix) == 0:  # If a landmark vanishes because of the affine transformations
                return None   # applied to the mask, we skip this instance

            y_coord.extend([int(np.mean(iy)), int(np.mean(ix))])

        return np.array(y_coord)

    def get_y_mask(self, image: np.ndarray, landmark_coord: np.ndarray):
        y_mask = np.zeros((image.shape[0], image.shape[1], 1))

        for landmark_i, (x, y) in enumerate(landmark_coord):
            y_mask[y, x] = landmark_i + 1

        return y_mask

    def get_img_with_mask(self, image: np.ndarray, landmark_coord: np.ndarray):
        self.mask_channel = image.shape[-1]
        self.nbr_landmarks = landmark_coord.shape[1]

        img_with_mask = []

        for image_i, landmark_i in zip(image, landmark_coord):
            mask_i = self.get_y_mask(image_i, landmark_i)
            img_with_mask_i = np.dstack([image_i, mask_i])
            img_with_mask.append(img_with_mask_i)

        return np.array(img_with_mask)


def grid_detection(X, y):
    """
    Function used to train a CNN model to recognize grid and output its corners' coordinates.
    :param X: array of shape (N, X, Y, 1) containing the grids. Can be obtained with load_grids()
    :param y: array of shape (N, 4, 2) containing the ground-truth coordinates for the corners. Can be obtained
    with load_corners_coordinates()
    :return: np.ndarray - Array with the ground-truth labels for the test set;
    np.ndarray - Array with the predicted labels for the test set; model
    """

    datagen_args = dict(
        samplewise_center=False,
        samplewise_std_normalization=False,
        rotation_range=0,
        width_shift_range=2,
        height_shift_range=2,
        shear_range=0,
        horizontal_flip=False,
        vertical_flip=False)
    datagenerator = ImageDataGenerator_landmarks(datagen_args, 'standardization')

    nbr_epochs = 50

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)

    Xy_train = datagenerator.get_img_with_mask(x_train, y_train)

    x_test -= np.mean(x_test)
    x_test /= np.std(x_test)

    y_test = y_test.reshape((y_test.shape[0], -1))

    input_dim = x_train[0].shape
    input_grid = Input(shape=input_dim)
    output_dim = datagenerator.nbr_landmarks * 2
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
    model = Model(input_grid, x)
    model.compile(optimizer='Adadelta', loss='mean_squared_error')
    model.fit_generator(datagenerator.flow(Xy_train, batch_size=16),
                        steps_per_epoch=len(x_train) / 32,
                        epochs=nbr_epochs,
                        shuffle=True,
                        validation_data=(x_test, y_test),
                        verbose=1)
    y_pred = model.predict(x_test)

    return y_test, y_pred, model






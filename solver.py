import constraint as clp
import numpy as np
import time

def solveSudoku(grid, unknownVariableSymbol=0):
    """
    Function to solve a sudoku grid with constraint logic programming.
    Only supports classic sudoku grids.
    :param grid: (np.array(9,9)) the sudoku grid to be solved.
    :param unknownVariableSymbol: (str, or 0) the symbol that was used to indicate an unknown value in the sudoku grid.
    :return: a numpy array of shape (9,9), i.e. the solved sudoku grid.
    """

    # Sanity checks
    assert type(grid) == np.ndarray, '\'grid\' should be a numpy.ndarray of shape (9,9)'
    assert grid.shape == (9, 9), '\'grid\' should be a numpy.ndarray of shape (9,9)'
    if type(unknownVariableSymbol) == int:
        assert not 1 <= unknownVariableSymbol <= 9, '\'unknownVariableSymbol\' should not be between 1 and 9'

    tic = time.time()

    if unknownVariableSymbol != 0:
        grid[grid == unknownVariableSymbol] = 0
        grid = grid.astype(int)

    sudoku = clp.Problem()

    # Introducing 81 variables. Each variable is represented by a number between [1; 81] and
    # can take a value between [1; 9].
    variablesGrid = np.arange(1, 82).reshape((9, 9))
    possibleValues = np.arange(1, 10, dtype=int)
    sudoku.addVariables(variablesGrid.flatten(), possibleValues.tolist())

    # Fix values for known variables
    knownVariables = np.nonzero(grid)
    for i, j in zip(knownVariables[0], knownVariables[1]):
        sudoku._variables.pop(variablesGrid[i][j])
        sudoku.addVariable(variablesGrid[i][j], [grid[i][j]])

    # Adding constraints for each row
    for row in variablesGrid:
        sudoku.addConstraint(clp.AllDifferentConstraint(), row.tolist())

    # Adding constraints for each column
    for col in variablesGrid.T:
        sudoku.addConstraint(clp.AllDifferentConstraint(), col.tolist())

    # Adding constraints for each sub-grid
    for i in range(0, 7, 3):
        for j in range(0, 7, 3):
            sudoku.addConstraint(clp.AllDifferentConstraint(), variablesGrid[i:i+3, j:j+3].flatten().tolist())

    # The solution is returned as a dictionary where each (key, value) pair is a (Variable, Value) pair
    solutionDic = sudoku.getSolution()
    solution = [solutionDic[key] for key in variablesGrid.flatten()]

    toc = time.time()
    elapsed_time = toc - tic

    return np.reshape(solution, (9, 9)), elapsed_time

grid = [0,0,5,9,0,1,8,0,0,0,0,9,0,0,0,6,0,0,0,8,0,0,2,0,0,9,0,0,9,0,0,7,0,0,4,0,8,0,3,4,0,5,7,0,2,0,7,0,0,1,0,0,3,0,0,1,0,0,4,0,0,5,0,0,0,8,0,0,0,4,0,0,0,0,7,2,0,3,1,0,0]
grid = np.array(grid).reshape((9,9))
print(grid)

solvedGrid, elapsed_time = solveSudoku(grid)
print(solvedGrid)
print(elapsed_time)
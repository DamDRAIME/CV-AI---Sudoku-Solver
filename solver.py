import constraint as clp
import numpy as np


def solveSudoku(grid, unknownVariableSymbol='0'):

    possibleValues = np.arange(1, 10, dtype=int)
    variablesGrid = np.arange(1, 82).reshape((9, 9))
    if unknownVariableSymbol != '0':
        grid[grid == unknownVariableSymbol] = 0
        grid = grid.astype(int)

    sudoku = clp.Problem()

    # Introducing 81 variables
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
    for i in range(0,7,3):
        for j in range(0,7,3):
            sudoku.addConstraint(clp.AllDifferentConstraint(), variablesGrid[i:i+3, j:j+3].flatten().tolist())

    solutionDic = sudoku.getSolution()
    solution = [solutionDic[key] for key in variablesGrid.flatten()]
    return np.reshape(solution, (9, 9))

grid = ['_','_',5,9,'_',1,8,'_','_','_','_',9,'_','_','_',6,'_','_','_',8,'_','_',2,'_','_',9,'_','_',9,'_','_',7,'_','_',4,'_',8,'_',3,4,'_',5,7,'_',2,'_',7,'_','_',1,'_','_',3,'_','_',1,'_','_',4,'_','_',5,'_','_','_',8,'_','_','_',4,'_','_','_','_',7,2,'_',3,1,'_','_']
grid = np.array(grid).reshape((9,9))

solvedGrid = solveSudoku(grid, unknownVariableSymbol='_')
print(solvedGrid)
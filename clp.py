class Problem:
    # TODO: Assert or not
    """
    Class used to define a problem and retrieve solutions
    """
    def __init__(self, verbose: int=1):
        self.variables = []
        self.constraints = []
        self.verbose = verbose

    def addVariables(self, variables: (list, int, str), domain: list):
        """
        Add all variables to the problem and assign them the same domain.

        :param variables: List/Int/String - Variable(s) that will be added to the problem
        :param domain: List - Domain that will be common to all variables
        :return: None
        """
        if type(variables) == int or type(variables) == str:
            variables = [variables]
        for var in variables:
            self.variables.append(Variable(var, domain))
        if self.verbose == 2:
            print('[+] Variables: {} were created with domain: {}.'.format(variables, domain))

    def getVariables(self, variables: (list, str, int)):
        """
        "Getter" for a variable or a set of variable(s). Print back the variable(s) along with its/their domain(s).

        :param variable: List/Sting/Int - Variable(s) that will be retrieved
        :return: None
        """
        if type(variables) == int or type(variables) == str:
            variables = [variables]
        for variable in variables:
            exist, variableFound = self.__existVariable(variable)
            if exist:
                print('[.] Variable: {} has domain: {}.'.format(variableFound.name, variableFound.domain.domain))
            else:
                print('[.] Variable: {} was not found.'.format(variable))

    def removeFromDomain(self, variables: (list, int, str), values: (list, int)):
        if type(variables) == int or type(variables) == str:
            variables = [variables]
        if type(values) == int:
            values = [values]
        variablesId = []
        for variable in variables:
            exist, variableFound = self.__existVariable(variable)
            if not exist:
                print('[!] Variable: {} was not found. Create it first with addVariable() method.'.format(variable))
                break
            variablesId.append(variableFound)
        else:
            for var in variablesId:
                var.domain.discardValuesFromDomain(values)
            if self.verbose == 2:
                print('[-] Variable(s): {} had their domains adjusted'.format(variables))

    def __existVariables(self, variables: (list, str, int)):
        variablesId =[]
        for variable in variables:
            exist, variableFound = self.__existVariable(variable)
            if not exist:
                print('[!] Variable: {} was not found. Create it first with addVariable() method.'.format(variable))
                return None
            variablesId.append(variableFound)
        return variablesId

    def __existVariable(self, variable: (str, int)):
        for var in self.variables:
            if var.name == str(variable):
                return True, var
        return False, None

    def getConstraintDescription(self, constraint: str):
        """
        "Getter" for a constraint. Print back the constraint description.

        :param constraint: Sting - Name of the constraint that will be retrieved
        :return: None
        """
        dictConstraints = {'alldifferent': AllDifferent(None, None),
                           'greaterthan': GreaterThan(None, None),
                           'lessthan': LessThan(None, None)}
        dummyConstraint = dictConstraints.get(constraint.lower(), None)
        if dummyConstraint is None:
            print('[.] Constraint: {} was not found.'.format(constraint))
        else:
            dummyConstraint.getDescription()

    def getConstraints(self):
        for constraint in self.constraints:
            variables = [var.name for var in constraint.variables]
            print('[.] Problem includes constraint: {} for variable(s): {}'.format(constraint.name, variables))

    def allDifferent(self, variables: list):
        assert (type(variables) == list and len(variables) >= 2), '[!] More than one variable should be submitted.'
        variablesId = self.__existVariables(variables)
        if variablesId is not None:
            for i, variable1 in enumerate(variablesId[:-1]):
                for variable2 in variablesId[i+1:]:
                    self.constraints.append(AllDifferent([variable1, variable2]))
            if self.verbose == 2:
                print('[+] Constraint: AllDifferent was added on the following variables : {}.'.format(variables))

    def differentThan(self, variables: list, variableComp: (int,str)):
        if type(variables) == int or type(variables) == str:
            variables = [variables]
        variables.append(variableComp)
        variablesId = self.__existVariables(variables)
        if variablesId is not None:
            variableCompId = variablesId[-1]
            for variable in variablesId[:-1]:
                self.constraints.append(DifferentThan(variable, variableCompId))
            if self.verbose == 2:
                print('[+] Constraint: AllDifferent was added on the following variables : {}.'.format(variables[:-1]))

    def greaterThan(self, variables: (list, int, str), variableComp: (int,str)):
        assert type(variableComp) == int or type(variableComp) == str, \
            '[!] Only one variable is allowed for comparison.'
        variablesId = []
        if type(variables) == int or type(variables) == str:
            variables = [variables]
        variables.append(variableComp)
        for variable in variables:
            exist, variableFound = self.__existVariable(variable)
            if not exist:
                print('[!] Variable: {} was not found. Create it first with addVariable() method.'.format(variable))
                break
            variablesId.append(variableFound)
        else:
            self.constraints.append(GreaterThan(variablesId[:-1], variablesId[-1]))
            if self.verbose == 2:
                print('[+] Constraint: GreaterThan was added on the following variables : {}.'.format(variables[:-1]))

    def lessThan(self, variables: (list, int, str), variableComp: (int,str)):
        assert type(variableComp) == int or type(variableComp) == str, \
            '[!] Only one variable is allowed for comparison.'
        variablesId = []
        if type(variables) == int or type(variables) == str:
            variables = [variables]
        variables.append(variableComp)
        for variable in variables:
            exist, variableFound = self.__existVariable(variable)
            if not exist:
                print('[!] Variable: {} was not found. Create it first with addVariable() method.'.format(variable))
                break
            variablesId.append(variableFound)
        else:
            self.constraints.append(LessThan(variablesId[:-1], variablesId[-1]))
            if self.verbose == 2:
                print('[+] Constraint: LessThan was added on the following variables : {}.'.format(variables[:-1]))


class Variable(object):
    """
    Class used to store the variables of a problem
    """
    def __init__(self, name, domain):
        self.name = str(name)
        self.domain = Domain(domain)
        self.value = None

    def assignValue(self, value):
        self.value = value


class Domain(object):
    """
    Class used to store the possible values of a variable
    """
    def __init__(self, domain):
        self.domain = set(domain)

    def discardValuesFromDomain(self, values):
        for value in values:
            self.domain.discard(value)


class Constraint(object):
    """
    Class used to store the constraints of a problem
    """
    def __init__(self, name, description):
        self.name = str(name)
        self.description = description

    def getDescription(self):
        print('[.] Description of constraint: {}'.format(self.name))
        print(self.description)


class Different(Constraint):
    def __init__(self, name, description, variables):
        self.variables = variables
        Constraint.__init__(self, name, description)

    def checkConstraint(self):
        return self.variables[0].value != self.variables[0].value


class AllDifferent(Different):
    def __init__(self, variables):
        name = 'AllDifferent'
        description = 'This constraint enforces that all the variables involved will be different from one another'
        Different.__init__(self, name, description, variables)


class DifferentThan(Constraint):
    def __init__(self, variable, variableComp):
        name = 'DifferentThan'
        description = 'This constraint enforces that all the variables involved will be different from another variable.'
        variables = [variable, variableComp]
        Different.__init__(self, name, description, variables)


class GreaterThan(Constraint):
    def __init__(self, variables, variableComp):
        name = 'GreaterThan'
        description = 'This constraint enforces that some variable(s) should be greater than another variable.\n' \
                      'It assumes that variables\' domains can be compared.'
        Constraint.__init__(self, name, description)
        self.variables = variables
        self.variableComp = variableComp

    def checkConstraint(self):
        varVal = [variable.value for variable in self.variables]
        for vv in varVal:
            if vv <= self.variableComp.value:
                return False
        return True


class LessThan(Constraint):
    def __init__(self, variables, variableComp):
        name = 'LessThan'
        description = 'This constraint enforces that some variable(s) should be less than another variable.\n' \
                      'It assumes that variables\' domains can be compared.'
        Constraint.__init__(self, name, description)
        self.variables = variables
        self.variableComp = variableComp

    def checkConstraint(self):
        varVal = [variable.value for variable in self.variables]
        for vv in varVal:
            if vv >= self.variableComp.value:
                return False
        return True


class Solver(object):
    def __init__(self):
        pass




sudo = Problem(verbose=2)
sudo.addVariables('A', [1, 2, 3, 4])
sudo.addVariables(['B', 'C', 'D'], [1, 2, 3, 4])
sudo.allDifferent(['A', 'B', 'C'])
sudo.differentThan(['A', 'C'], 'D')
sudo.greaterThan('C', 'D')
sudo.removeFromDomain('B', 1)
sudo.removeFromDomain('C', 3)
sudo.getConstraints()
sudo.getVariables(['A', 'B', 'C', 'D'])

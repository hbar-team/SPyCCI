class Engine:
    """
    Simple base class for the definition of a software engine. The class sets the `level_of_theory`
    and `method` attributes.

    Arguments
    ---------
    method: str
        The string indicating the method to be used in the calculation.
    """
    def __init__(self, method: str) -> None:
        self.method = method
        self.level_of_theory = f"{self.__class__.__name__} || method: {self.method}"

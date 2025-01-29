from typing import Callable
import numpy as np


class Dual:
    """
    A class for representing dual numbers for automatic differentiation

    Each dual number has the form a + bε, where a is the real part and b is coefficient to ε, where ε ** 2 = 0.

    ε represents a very small number.

    Attributes:
        real: (float): Real part
        dual: (float): Dual part
    """
    

    def __init__(self, real: float, dual: float = 0.0):
        """Input the real and dual parts of the dual number.

        Args:
            real (float): Real part of the dual number.
            dual (float, optional): Dual part of the dual number. Defaults to 0.0.
        """
        self.real = real
        self.dual = dual

    # THEORY: https://fs.unm.edu/DualNumbers.pdf

    # Overloading the arithmetic operators
    # Addition
    def __add__(self, other) -> "Dual":
        """Overload + operator to perform addition involving Dual Numbers

        Args:
            other (int, float, Dual): The right hand side of the + operator to be added to the Dual number

        Returns:
            Dual: Resulting sum of the addition
        """
        # Check if the other object is a Dual number
        if not isinstance(other, Dual):
            # If the other number is not a Dual number, make it a Dual number
            other = Dual(other)
        return Dual(self.real + other.real, self.dual + other.dual)

    # Reverse addition (if right side of operand is a Dual number)
    def __radd__(self, other) -> "Dual":
        """Overload + operator to perform addition involving Dual numbers, when Dual number is on the right hand side of the operator.

        Args:
            other (int, float, Dual): The left hand side of the + operator to be added to the Dual number

        Returns:
            Dual: Resulting sum of the addition
        """
        # Since addition is commutative, we can just call the __add__ method
        return self.__add__(other)

    # Subtraction
    def __sub__(self, other) -> "Dual":
        """Overload - operator to perform subtraction involving Dual numbers.

        Args:
            other (int, float, Dual): Right hand side of - operator to be subtracted from the Dual number.

        Returns:
            Dual: Resulting difference of the subtraction
        """
        # Check if the other object is a Dual number
        if not isinstance(other, Dual):
            # Make it a Dual number
            other = Dual(other)
        return Dual(self.real - other.real, self.dual - other.dual)

    # Reverse subtraction (if right side of operand is a Dual number)
    def __rsub__(self, other) -> "Dual":
        """Overload - operator to perform subtraction involving Dual numbers, when Dual number is on the right hand side of the operator.

        Args:
            other (int, float, Dual): The left hand side of the - operator from which the Dual number is subtracted from

        Returns:
            Dual: Resulting difference of the subtraction
        """
        # Convert other number to Dual number if needed
        if not isinstance(other, Dual):
            other = Dual(other)
        # Reverse the order (other - self)
        return Dual(other.real - self.real, other.dual - self.dual)

    # Multiplication
    def __mul__(self, other) -> "Dual":
        """Overload * operator to perform multiplication involving Dual numbers.

        Args:
            other (int, float, Dual): Right hand side of * operator to be multiplied to the Dual number.

        Returns:
            Dual: Resulting product of the multiplication
        """
        # Check if the other object is a Dual number
        if not isinstance(other, Dual):
            # Make it a Dual number
            other = Dual(other)
        # Following FOIL method (First, Outer, Inner, Last)
        return Dual(
            self.real * other.real, self.real * other.dual + self.dual * other.real
        )

    # Reverse multiplication (if right side of operand is a Dual number)
    def __rmul__(self, other) -> "Dual":
        """Overload * operator to perform multiplication involving Dual numbers, when the Dual number is on the right hand side of the operator.

        Args:
            other (int, float, Dual): Left hand side of * operator to be multiplied to the Dual number.

        Returns:
            Dual: Resulting product of the multiplication
        """
        # Since multiplication is commutative, we can just call the __mul__ method
        return self.__mul__(other)

    # Division
    def __truediv__(self, other) -> "Dual":
        """Overload / operator to perform division involving Dual numbers.

        Args:
            other (int, float, Dual): Right hand side of / operator to divide the Dual number.

        Returns:
            Dual: Resulting quotient of the division
        """
        # Check if the other object is a Dual number
        if not isinstance(other, Dual):
            # Make it a Dual number
            other = Dual(other)
            # Perform the division (https://fs.unm.edu/DualNumbers.pdf)
        real = self.real / other.real
        dual = (self.dual * other.real - self.real * other.dual) / (other.real**2)
        return Dual(real, dual)

    # Reverse division (if right side of operand is a Dual number)
    def __rtruediv__(self, other) -> "Dual":
        """Overload / operator to perform division involving Dual numbers, when the Dual number is on the right hand side of the operator.

        Args:
            other (int, float, Dual): Left hand side of / operator to be divided by the Dual number.

        Returns:
            Dual: Resulting quotient of the division
        """
        # Check if the other object is a Dual number
        if not isinstance(other, Dual):
            # Make it a Dual number
            other = Dual(other)
        # Perform the division (https://fs.unm.edu/DualNumbers.pdf) (order of operations is reversed)
        real = other.real / self.real
        dual = (other.dual * self.real - other.real * self.dual) / (self.real**2)
        return Dual(real, dual)

    # Power
    def __pow__(self, other) -> "Dual":
        """Overload ** operator to perform powers involving Dual numbers, when the Dual number is the base.

        Args:
            other (int, float, Dual): Right hand side of ** operator (exponent).

        Raises:
            ValueError: For when the real part is negative (logarithm of negative number is undefined)

        Returns:
            Dual: Resulting Dual number
        """
        # If the other is a dual number
        if isinstance(other, Dual):
            # (a + bε) ** (c + dε)
            a, b = self.real, self.dual
            c, d = other.real, other.dual
            # Raise error for negative a (as it is not valid for negative a)
            if a < 0:
                raise ValueError(
                    "The real part of base cannot be negative for exponents (undefined)"
                )
            # (a + bε) ** (c + dε) = a ** c * (1 + (b * c * log(a) + d * log(a) * ε)) - from Stack Exchange
            pow_real = a**c
            pow_dual = a ** (c - 1) * (b * c + a * d * np.log(a))
            return Dual(pow_real, pow_dual)
        else:
            # If other is a real number (a + bε) ** n
            a, b = self.real, self.dual
            pow_real = a**other
            pow_dual = other * b * (a ** (other - 1))
            return Dual(pow_real, pow_dual)

    # Reverse power
    def __rpow__(self, other) -> "Dual":
        """Overload ** operator to perform powers involving Dual numbers, when the Dual number is the exponent.

        Args:
            other (int, float, Dual): Left hand side of ** operator (base).

        Raises:
            ValueError: Prevent base being a non-numerical value.

        Returns:
            Dual: Resulting Dual number
        """
        # Check if other is either float or int
        if isinstance(other, (int, float)):
            # Make the other number a dual number
            # (a + bε) ** (c + dε)
            a, b = self.real, self.dual
            # Applying opposite principle to __pow__
            pow_real = other**a
            pow_dual = pow_real * b * np.log(other)
            return Dual(pow_real, pow_dual)
        else:
            # Raise a ValueError if the LHS is not a float or a int (Dual ** Dual handled by __pow__)
            raise ValueError(
                f"Unsupported operation between {type(other).__name__} and Dual"
            )

    # Create a representation function for interactive notebooks
    def __repr__(self) -> None:
        """Creates a representation of Dual numbers in ipython kernels.

        Returns:
            None: Prints a description of Dual number in notebook kernel output.
        """
        return f"Dual({self.real}, {self.dual})"

    # CLASS METHODS FOR DUAL NUMBERS
    # Derivative
    @classmethod
    def derivative(cls, func: Callable, x: float) -> float:
        """Function to be used for automatic differentiation, where the result is the derivative of the function evaluated at x
        Args:
            func (Callable): Function to be differentiated
            x (float): Value of x derivative is to be evaluated at

        Returns:
            float: Value of derivative evaluated at x
        """
        # Get the dual of the input x value
        dual_x = cls(x, 1.0)
        # Evaluate the function at x
        eval_x = func(dual_x)
        return eval_x.dual

    # IMPLEMENT COMMON FUNCTIONS f(x) AND GET DERIVATIVE f'(x)
    # sin(x)
    def sin(self) -> "Dual":
        """
        Returns sin(x), where x is a Dual number.

        x = a + b * ε
        sin(x) = sin(a) + b * cos(a) * ε

        Returns:
            Dual: returns the Dual representation of sin(x)
        """
        return Dual(np.sin(self.real), np.cos(self.real) * self.dual)

    # Compute the derivative of sin(x) directly
    @staticmethod  # staticmethod to directly evaluate the derivative
    def sin_derivative(x: float) -> float:
        """Direct static function to calculate the derivative of sin(x) at x

        Args:
            x (float): Value to evaluate the derivative

        Returns:
            float: Value of derivative evaluated at x
        """
        return Dual.derivative(np.sin, x)

    # cos(x)
    def cos(self) -> "Dual":
        """
        Returns cos(x), where x is a Dual number.

        x = a + b * ε
        cos(x) = cos(a) - b * sin(a) * ε

        Returns:
            Dual: returns the Dual representation of cos(x)
        """
        return Dual(np.cos(self.real), -np.sin(self.real) * self.dual)

    # Compute the derivative of cos(x) directly
    @staticmethod
    def cos_derivative(x: float) -> float:
        """Direct static function to calculate the derivative of cos(x) at x

        Args:
            x (float): Value to evaluate the derivative

        Returns:
            float: Value of derivative evaluated at x
        """
        return Dual.derivative(np.cos, x)

    # tan(x)
    def tan(self) -> "Dual":
        """
        Returns tan(x), where x is a Dual number.

        x = a + b * ε
        tan(x) = tan(a) + b * sec(a) ** 2 * ε = tan(a) + b * ((1 / cos(a)) ** 2) * ε

        Returns:
            Dual: returns the Dual representation of tan(x)
        """
        return Dual(np.tan(self.real), ((1 / np.cos(self.real)) ** 2) * self.dual)

    # Compute the derivative of tan(x) directly
    @staticmethod
    def tan_derivative(x: float) -> float:
        """Direct static function to calculate the derivative of tan(x) at x

        Args:
            x (float): Value to evaluate the derivative

        Returns:
            float: Value of derivative evaluated at x
        """
        return Dual.derivative(np.tan, x)

    # ln(x)
    def log(self) -> "Dual":
        """
        Returns ln(x), where x is a Dual number.

        x = a + b * ε
        ln(x) = ln(a) + (b / a) * ε

        Raises:
            ValueError: if the argument to ln is not positive

        Returns:
            Dual: returns the Dual representation of ln(x)
        """
        # Check for non-positive values
        if self.real <= 0:
            # Raise a Value error
            raise ValueError("The argument to ln must be positive.")
        return Dual(np.log(self.real), self.dual / self.real)

    # Compute the derivative of ln(x) directly
    @staticmethod
    def log_derivative(x: float) -> float:
        """Direct static function to calculate the derivative of ln(x) at x

        Args:
            x (float): Value to evaluate the derivative

        Returns:
            float: Value of derivative evaluated at x
        """
        return Dual.derivative(np.log, x)
    
    # exp(x)
    def exp(self) -> "Dual":
        """
        Returns exp(x), where x is a Dual number.

        x = a + b * ε
        exp(x) = exp(a) + b * exp(a) * ε

        Returns:
            Dual: returns the Dual representation of exp(x)
        """
        # Find the exponential of the real part
        exp_real = np.exp(self.real)
        return Dual(exp_real, exp_real * self.dual)

    @staticmethod
    def exp_derivative(x: float) -> float:
        """Direct static function to calculate the derivative of exp(x) at x

        Args:
            x (float): Value to evaluate the derivative

        Returns:
            float: Value of derivative evaluated at x
        """
        return Dual.derivative(np.exp, x)

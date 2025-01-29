# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, tan, log, exp

cdef class Dual:
    """
    A Cython implementation of the Dual number class for automatic differentiation
    """
    cdef public double real
    cdef public double dual
    
    def __init__(self, double real, double dual=0.0):
        self.real = real
        self.dual = dual
    
    def __add__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        return Dual(self.real + other.real, self.dual + other.dual)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        return Dual(self.real - other.real, self.dual - other.dual)
    
    def __rsub__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        return Dual(other.real - self.real, other.dual - self.dual)
    
    def __mul__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        return Dual(
            self.real * other.real,
            self.real * other.dual + self.dual * other.real
        )
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        cdef double real = self.real / other.real
        cdef double dual = (self.dual * other.real - self.real * other.dual) / (other.real * other.real)
        return Dual(real, dual)
    
    def __rtruediv__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        cdef double real = other.real / self.real
        cdef double dual = (other.dual * self.real - other.real * self.dual) / (self.real * self.real)
        return Dual(real, dual)
    
    def __pow__(self, other):
        cdef double a, b, c, d
        cdef double pow_real, pow_dual
        
        if isinstance(other, Dual):
            if self.real < 0:
                raise ValueError("The real part of base cannot be negative for exponents")
            a = self.real
            b = self.dual
            c = other.real
            d = other.dual
            pow_real = a ** c
            pow_dual = a ** (c - 1) * (b * c + a * d * log(a))
            return Dual(pow_real, pow_dual)
        else:
            a = self.real
            b = self.dual
            pow_real = a ** other
            pow_dual = other * b * (a ** (other - 1))
            return Dual(pow_real, pow_dual)
    
    def __repr__(self):
        return f"Dual({self.real}, {self.dual})"
    
    @classmethod
    def derivative(cls, func, double x):
        cdef Dual dual_x = cls(x, 1.0)
        cdef Dual eval_x = func(dual_x)
        return eval_x.dual
    
    cpdef Dual sin(self):
        return Dual(sin(self.real), cos(self.real) * self.dual)
    
    @staticmethod
    def sin_derivative(double x):
        return Dual.derivative(np.sin, x)
    
    cpdef Dual cos(self):
        return Dual(cos(self.real), -sin(self.real) * self.dual)
    
    @staticmethod
    def cos_derivative(double x):
        return Dual.derivative(np.cos, x)
    
    cpdef Dual tan(self):
        cdef double cos_x = cos(self.real)
        return Dual(tan(self.real), (1.0 / (cos_x * cos_x)) * self.dual)
    
    @staticmethod
    def tan_derivative(double x):
        return Dual.derivative(np.tan, x)
    
    cpdef Dual log(self):
        if self.real <= 0:
            raise ValueError("The argument to ln must be positive.")
        return Dual(log(self.real), self.dual / self.real)
    
    @staticmethod
    def log_derivative(double x):
        return Dual.derivative(np.log, x)
    
    cpdef Dual exp(self):
        cdef double exp_real = exp(self.real)
        return Dual(exp_real, exp_real * self.dual)
    
    @staticmethod
    def exp_derivative(double x):
        return Dual.derivative(np.exp, x)
    
    
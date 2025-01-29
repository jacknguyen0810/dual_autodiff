import pytest
import numpy as np
from dual_autodiff.dual import Dual

    
def test_init() -> None:
    dual = Dual(1, 2)
    # Test that the object is an instance of Dual
    assert isinstance(dual, Dual)
    # Check dual components are being generated correctly
    assert dual.real == 1
    assert dual.dual == 2
    
def test_dual_real_only() -> None:
    dual = Dual(1, 0)
    assert dual.real == 1
    assert dual.dual == 0

def test_dual_dual_only() -> None:
    dual = Dual(0, 1)
    assert dual.real == 0
    assert dual.dual == 1
    
def test_dual_add_dual() -> None:
    dual1 = Dual(3, 4)
    dual2 = Dual(5, 6)
    addition = dual1 + dual2
    assert addition.real == 8
    assert addition.dual == 10
    
def test_dual_add_real() -> None:
    dual = Dual(1, 2)
    num = 5
    addition = dual + num
    assert addition.real == 6
    assert addition.dual == 2
    
def test_dual_radd() -> None:
    num = 5
    dual = Dual(1, 2)
    addition = num + dual
    assert addition.real == 6
    assert addition.dual == 2
    
def test_dual_sub_dual() -> None:
    dual1 = Dual(5, 6)
    dual2 = Dual(3, 4)
    addition = dual1 - dual2
    assert addition.real == 2
    assert addition.dual == 2
    
def test_dual_sub_real() -> None:
    dual = Dual(1, 2)
    num = 5
    sub = dual - num
    assert sub.real == -4
    assert sub.dual == 2
    
def test_dual_rsub() -> None:
    num = 5
    dual = Dual(1, 2)
    sub = num - dual
    assert sub.real == 4
    assert sub.dual == -2
    
def test_dual_mul() -> None:
    dual1 = Dual(2, 3)
    dual2 = Dual(4, 5)
    prod = dual1 * dual2
    assert prod.real == 8
    assert prod.dual == 22

def test_dual_rmul() -> None:
    dual1 = Dual(2, 3)
    dual2 = Dual(4, 5)
    prod = dual2 * dual1
    assert prod.real == 8
    assert prod.dual == 22
    
def test_dual_mul_int() -> None:
    dual = Dual(2, 3)
    num = 4
    prod = dual * num
    assert prod.real == 8
    assert prod.dual == 12
    
def test_dual_rmul_int() -> None:
    dual = Dual(2, 3)
    num = 4
    prod = num * dual
    assert prod.real == 8
    assert prod.dual == 12
    
def test_dual_div() -> None:
    dual1 = Dual(3, 4)
    dual2 = Dual(1, 2)
    div = dual1 / dual2
    assert isinstance(div, Dual)
    assert div.real == 3
    assert div.dual == -2
    
def test_dual_div_num() -> None:
    dual = Dual(3, 4)
    num = 4
    div = dual / num
    assert div.real == 0.75
    assert div.dual == 1
    
def test_dual_rdiv() -> None:
    dual = Dual(3, 4)
    num = 4
    div = num / dual
    assert isinstance(div, Dual)
    assert div.real == (4 / 3)
    assert div.dual == (-16 / 9)

def test_dual_pow() -> None:
    base = Dual(1, 2)
    exponent = Dual(3, 4)
    power = base ** exponent
    assert power.real == 1
    assert power.dual == 6
    
def test_dual_pow_num() -> None:
    base = Dual(1, 2)
    exponent = 2
    power = base ** exponent
    assert power.real == 1
    assert power.dual == 4
    
def test_dual_rpow_num() -> None:
    base = 2
    exponent = Dual(1, 2)
    power = base ** exponent
    print(power.dual)
    assert power.real == 2
    assert power.dual == 4 * np.log(2)
    
def test_dual_derivative() -> None:
    # TODO: 
    def f(x):
        return 2 * x**2 + 2 * x + 2
    
    dual = Dual(2, 1)
    d_dx = dual.derivative(f, 1)
    
    assert d_dx == 6

def test_dual_sin() -> None:
    dual = Dual(0, np.pi)
    dual_sin = dual.sin()
    assert dual_sin.real == 0
    assert dual_sin.dual == np.pi
    
def test_dual_sin_derivative() -> None:
    dual = Dual(1, 1)
    assert dual.sin_derivative(np.pi) == -1

def test_dual_cos() -> None:
    dual = Dual(np.pi, 1)
    dual_cos = dual.cos()
    
    assert dual_cos.real == -1
    assert pytest.approx(dual_cos.dual) == 0        # Approx to prevent floating point error
    
def test_dual_cos_derivative() -> None:
    dual = Dual(1, 1)
    assert dual.cos_derivative(0) == 0
    
def test_dual_tan() -> None:
    dual = Dual(np.pi, 1)
    dual_tan = dual.tan()
    assert pytest.approx(dual_tan.real) == 0
    assert dual_tan.dual == 1
    
def test_dual_log() -> None:
    dual = Dual(2, 2)
    dual_log = dual.log()
    
    assert dual_log.real == np.log(2)
    assert dual_log.dual == 1
    
def test_dual_log_derivative() -> None:
    dual = Dual(1, 1)
    assert dual.log_derivative(2) == 0.5
    
def test_dual_exp() -> None:
    dual = Dual(2, 3)
    dual_exp = dual.exp()
    
    assert dual_exp.real == np.exp(2)
    assert dual_exp.dual == (3 * np.exp(2))
    
def test_dual_exp_derivative() -> None:
    dual = Dual(1, 1)
    assert dual.exp_derivative(2) == np.exp(2)


    
if __name__ == "__main__":
    pytest.main()
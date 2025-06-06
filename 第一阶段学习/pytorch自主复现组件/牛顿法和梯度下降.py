'''求解根号2'''
import numpy as np
def fx(x, a):
    return (x**2 - a)**2

def dx(x, a):
    return 2*(x**2 - a)*2*x

def newton_pre(x, a, esp):
    error = 1
    while(error > esp):
        x_new = x - fx(x, a)/dx(x, a)
        error = np.abs(x_new - x)
        x = x_new
    return x_new
    
def gradient_descent(x, a, esp, lr):
    error = 1
    while(error > esp):
        x_new = x - lr*dx(x, a)
        error = np.abs(x_new - x)
        x = x_new
        
    return x_new

if "__main__" == '__main__':
    a = 2
    x = 10
    esp = 1e-8
    lr = 1e-3
    x1_t = newton_pre(x, a, esp)
    x2_t = gradient_descent(x, a, esp, lr)
    print(f"gradient_descent: 根号2:{x2_t}")
    print(f"newton_pre: 根号2:{x1_t}")
    pass


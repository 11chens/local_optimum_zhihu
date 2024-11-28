import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x)

def f_prime(x):
    return np.cos(x)

def f_double_prime(x):
    return -np.sin(x)

def f_triple_prime(x):
    return -np.cos(x)

def f_fourth_prime(x):
    return np.sin(x)

def f_fifth_prime(x):
    return np.cos(x)

def f_sixth_prime(x):
    return -np.sin(x)

def f_seventh_prime(x):
    return -np.cos(x)

def taylor_1(x, a):
    return f(a) + f_prime(a) * (x - a)

def taylor_3(x, a):
    return f(a) + f_prime(a) * (x - a) + 0.5 * f_double_prime(a) * (x - a)**2 + (1/6) * f_triple_prime(a) * (x - a)**3

def taylor_5(x, a):
    return f(a) + f_prime(a) * (x - a) + 0.5 * f_double_prime(a) * (x - a)**2 + (1/6) * f_triple_prime(a) * (x - a)**3 + (1/24) * f_fourth_prime(a) * (x - a)**4 + (1/120) * f_fifth_prime(a) * (x - a)**5

def taylor_7(x, a):
    return f(a) + f_prime(a) * (x - a) + 0.5 * f_double_prime(a) * (x - a)**2 + (1/6) * f_triple_prime(a) * (x - a)**3 + (1/24) * f_fourth_prime(a) * (x - a)**4 + (1/120) * f_fifth_prime(a) * (x - a)**5 + (1/720) * f_sixth_prime(a) * (x - a)**6 + (1/5040) * f_seventh_prime(a) * (x - a)**7

x = np.linspace(-4*np.pi, 4*np.pi, 400)

a = 0

y_1 = taylor_1(x, a)
y_3 = taylor_3(x, a)
y_5 = taylor_5(x, a)
y_7 = taylor_7(x, a)


y = f(x)

plt.figure(figsize=(10, 6))  


plt.plot(x, y, label="$f(x) = sin(x)$", color="blue", lw=3)

plt.plot(x, y_1, label="1st order Taylor", linestyle="--", color="green", lw=2)
plt.plot(x, y_3, label="3rd order Taylor", linestyle="--", color="purple", lw=2)
plt.plot(x, y_5, label="5th order Taylor", linestyle="--", color="brown", lw=2)
plt.plot(x, y_7, label="7th order Taylor", linestyle="--", color="red", lw=2)

plt.title('Taylor Expansions of $f(x) = sin(x)$ at $x = 0$', fontsize=16, fontweight='bold')
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)


plt.legend(loc='best', fontsize=12)
plt.ylim(-2, 2)
plt.xlim(-4*np.pi, 4*np.pi)

plt.grid(True, linestyle=':', color='gray', alpha=0.7)

plt.tick_params(axis='both', which='major', labelsize=12)

def save_plot(filename):
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")

save_plot("taylor_expansion_sin.png")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os

ALPHA = 0.12

# Newton's method
def rosenbrock_scaled(x, y, c=0.025):
    """Scaled Rosenbrock function"""
    return c * ((1 - x)**2 + 100 * (y - x**2)**2)

# Gradient of the Rosenbrock function
def rosenbrock_grad(x, y, c=0.025):
    """Gradient of the scaled Rosenbrock function"""
    dfdx = c * (-2 * (1 - x) - 400 * x * (y - x**2))
    dfdy = c * 200 * (y - x**2)
    return np.array([dfdx, dfdy])

# Hessian of the Rosenbrock function
def rosenbrock_hessian(x, y, c=0.025):
    """Hessian of the scaled Rosenbrock function"""
    d2fdx2 = c * (2 - 400 * (y - x**2) + 1200 * x**2)
    d2fdy2 = c * 200
    d2fdxdy = c * (-400 * x)
    return np.array([[d2fdx2, d2fdxdy], [d2fdxdy, d2fdy2]])

def steepest_descent(f, gradient, x0, max_iter=100, tol=1e-6):
    x = x0
    path = [x]
    alpha = ALPHA

    for i in range(max_iter):
        grad = gradient(x[0], x[1])
        
        # Perform line search to find the optimal step size
        
        # Update x with the optimal step size
        alpha = alpha * 0.99
        if alpha < 0.1:
            alpha = 0.1
        x_new = x - alpha * grad
        path.append(x_new)
        
        # If the change is small enough, stop
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    
    return np.array(path)

def newtons_method(f, gradient, hessian, x0, max_iter=100, tol=1e-6):
    x = x0
    path = [x]
    
    for i in range(max_iter):
        grad = gradient(x[0], x[1])
        H = hessian(x[0], x[1])

        # delta_x = H_inv * grad
        try:
            H_inv = np.linalg.inv(H) 
        except np.linalg.LinAlgError:
            print(f"Warning: Hessian is singular at iteration {i+1}.")
            break
        
        delta_x = H_inv @ grad
        
        x_new = x - delta_x
        path.append(x_new)
        
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    
    return np.array(path)

x0 = np.array([-0.8, -1.5])
steep_path = steepest_descent(rosenbrock_scaled, rosenbrock_grad, x0)
newton_path = newtons_method(rosenbrock_scaled, rosenbrock_grad, rosenbrock_hessian, x0)

# Create a meshgrid for plotting the 3D surface
x = np.linspace(-3, 3, 400)
y = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x, y)


Z = rosenbrock_scaled(X, Y)


temp_folder = "temp_frames"
os.makedirs(temp_folder, exist_ok=True)


gif_path = "steepest_vs_newton_method_rotation.gif"

fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)

ax1.plot(steep_path[:, 0], steep_path[:, 1], rosenbrock_scaled(steep_path[:, 0], steep_path[:, 1]), 'ro-', lw=2, markersize=8)
ax1.set_title("Steepest Descent Method")

ax2.plot(newton_path[:, 0], newton_path[:, 1], rosenbrock_scaled(newton_path[:, 0], newton_path[:, 1]), 'bo-', lw=2, markersize=8)
ax2.set_title("Newton's Method")

cnt = 0
for ax in [ax1, ax2]:
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X, Y)')

i = 1
ang_idx = 0
while(i < len(steep_path)):
    ax1.cla()  
    ax2.cla() 
    

    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
    ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
    

    ax1.plot(steep_path[:i, 0], steep_path[:i, 1], rosenbrock_scaled(steep_path[:i, 0], steep_path[:i, 1]), 'ro-', lw=2, markersize=8)
    ax1.set_title("Steepest Descent Method")

    ax2.plot(newton_path[:i, 0], newton_path[:i, 1], rosenbrock_scaled(newton_path[:i, 0], newton_path[:i, 1]), 'bo-', lw=2, markersize=8)
    ax2.set_title("Newton's Method")
    

    for ax in [ax1, ax2]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('f(X, Y)')
    

    angle = (ang_idx * 6) % 360  
    ax1.view_init(elev=30, azim=angle)
    ax2.view_init(elev=30, azim=angle)
    
    frame_path = os.path.join(temp_folder, f"frame_{i}_{cnt}.png")
    plt.savefig(frame_path)
    frame = imageio.imread(frame_path)
    ang_idx += 1

    cnt = (cnt+1)%3 
    if cnt == 0:
        i += 1




# for frame in os.listdir(temp_folder):
#     os.remove(os.path.join(temp_folder, frame))
# os.rmdir(temp_folder)

print(f"GIF saved to {gif_path}")

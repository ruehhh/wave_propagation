import numpy as np
import math as m
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Domain size
w = h = 10
# Setting mesh
dx = dy = .05
nx, ny = int(w/dx), int(h/dy)
dx2, dy2 = dx*dx, dy*dy
# Propagation speed
c = 1
# dt
dt = (dx*dy)/c*np.sqrt(1/(dx2+dy2))
dt2 = dt*dt


# Initial conditions
def f(x, y):
    # Insert initial condition here, should satisfy Dirichlet BCs
    return m.exp(- (x-w/2)**2 - (y-h/2)**2)


u = np.zeros((ny, nx))
v = u.copy()
# Fills in the grid with the chosen initial condition, the boundary values are left as 0
for i in range(1, ny-1):
    for j in range(1, nx-1):
        u[i, j] = f(j*dx, i*dy)


# Define Laplacian and sum of laplacians for integral form of equation
def lap(u, dx, dy):
    u_ = u.copy()
    u_[1:-1, 1:-1] = (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1])/dx**2 + (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2])/dy**2
    return u_


integrated_laplacian = lap(u, dx, dy)


def evolve(u, v, integrated_laplacian):
    integrated_laplacian += lap(u, dx, dy)
    v = u + c**2 * dt2 * integrated_laplacian
    u = v.copy()
    return u, v, integrated_laplacian


# Set up plot
fig = plt.figure()
ax = fig.add_subplot()
im = ax.imshow(u,cmap=plt.get_cmap('twilight'), vmin=-1, vmax=1, interpolation='bicubic')
ax.set_axis_off()
ax.set_title('t=0')


# Animate
def animate(i):
    global u, v, integrated_laplacian
    u, v, integrated_laplacian = evolve(u, v, integrated_laplacian)
    ax.set_title('t = {:.1f}'.format(i*dt))
    im.set_data(u.copy())


def create_animation(n_steps, filename=None):
    ani = anim.FuncAnimation(fig, animate, frames = n_steps, repeat=False, interval=1)
    writer = anim.PillowWriter(fps=15, metadata=dict(artist='ruehhh'), bitrate=1800)
    if filename:
        ani.save(filename=filename, writer=writer)
    plt.show()


if __name__ == "__main__":
    create_animation(500)

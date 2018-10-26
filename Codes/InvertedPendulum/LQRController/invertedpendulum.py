import os
from math import sin, cos, pi
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1, 0.009991, 0.0001336, 4.453e-07],
			  [0, 0.9982, 0.02672, 0.0001336],
			  [0, -2.272e-05, 1.002, 0.01001],
			  [0, -0.004544, 0.3119, 1.002]])


B = np.array([[9.086e-05], 
			 [0.01817], 
			 [0.0002272], 
			 [0.04544]])
 
C = np.array([[1, 0, 0, 0],
			  [0, 0, 1, 0]])

D = np.array([0, 0]).reshape(-1,1)

# K = np.array([-61.9933, -33.5040, 95.0597, 18.8300]).reshape(1,-1)
K = np.array([-0.9384, -1.5656, 18.0351, 3.3368]).reshape(1,-1)

t_span = np.arange(0, 10, 0.01);
x0 = np.array([0, 0, 0, 0]).reshape(-1,1)
u = 0.5*np.ones(t_span.size); 

u[50:60] = 1;
u[200:150] = -1.5;
u[300:310] = 2;
u[450:460] = -3;

x_out = np.zeros([x0.size, t_span.size]);
for t in range(0, t_span.size-1):
	x_t = np.matmul(A-np.matmul(B, K), x_out[:, t].reshape(-1,1)) + (B * u[t])
	x_out[:,t+1] = x_t[:,0]


# plt.figure(1)

# plt.subplot(1,2,1)
# plt.plot(t_span, x_out[0,:])

# plt.subplot(1,2,2)
# plt.plot(t_span, x_out[2,:])

# plt.show()

data = np.vstack([t_span, x_out]); data = data.T

fig = plt.figure(0)
fig.suptitle("Pendulum on Cart")

cart_time_line = plt.subplot2grid(
    (12, 12),
    (9, 0),
    colspan=12,
    rowspan=3
)
cart_time_line.axis([
    0,
    10,
    min(data[:,1])*1.1-2,
    max(data[:,1])*1.1+2,
])
cart_time_line.set_xlabel('time (s)')
cart_time_line.set_ylabel('x (m)')
cart_time_line.plot(data[:,0], data[:,1],'r-')

pendulum_time_line = cart_time_line.twinx()
pendulum_time_line.axis([
    0,
    10,
    min(data[:,3])*1.1-.1,
    max(data[:,3])*1.1
])
pendulum_time_line.set_ylabel('theta (rad)')
pendulum_time_line.plot(data[:,0], data[:,3],'g-')

cart_plot = plt.subplot2grid(
    (12,12),
    (0,0),
    rowspan=8,
    colspan=12
)
cart_plot.axes.get_yaxis().set_visible(False)

time_bar, = cart_time_line.plot([0,0], [10, -10], lw=3)
def draw_point(point):
    time_bar.set_xdata([t, t])
    cart_plot.cla()
    cart_plot.axis([-1.1,.1,-.5,.5])
    cart_plot.plot([point[1]-.1,point[1]+.1],[0,0],'r-',lw=5)
    cart_plot.plot([point[1],point[1]+.4*sin(point[3])],[0,.4*cos(point[3])],'g-', lw=4)
t = 0
fps = 25.
frame_number = 1
for point in data:
    if point[0] >= t + 1./fps or not t:
        draw_point(point)
        t = point[0]
        fig.savefig('img/_tmp%03d.png' % frame_number)
        frame_number += 1

print(os.system("ffmpeg -framerate 25 -i img/_tmp%03d.png  -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4"))

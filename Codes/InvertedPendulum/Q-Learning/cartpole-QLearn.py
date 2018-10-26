# Inspired by https://gist.github.com/n1try/af0b8476ae4106ec098fea1dfe57f578

import os
from math import sin, cos, pi
import gym
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import deque

class QCartPoleSolver():
    def __init__(self, buckets=(1, 1, 6, 12,), n_episodes=1000, n_win_ticks=500, min_alpha=0.1, min_epsilon=0.1, gamma=1.0, ada_divisor=25, max_env_steps=None, quiet=False, monitor=False):
        self.buckets = buckets # down-scaling feature space to discrete range
        self.n_episodes = n_episodes # training episodes 
        self.n_win_ticks = n_win_ticks # average ticks over 100 episodes required for win
        self.min_alpha = min_alpha # learning rate
        self.min_epsilon = min_epsilon # exploration rate
        self.gamma = gamma # discount factor
        self.ada_divisor = ada_divisor # only for development purposes
        self.quiet = quiet

        self.env = gym.make('CartPole-v0')
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        if monitor: self.env = gym.wrappers.Monitor(self.env, 'tmp/cartpole-1', force=True) # record results for upload

        self.Q = np.zeros(self.buckets + (self.env.action_space.n,))

    def discretize(self, obs):
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[state])

    def update_q(self, state_old, action, reward, state_new, alpha):
        self.Q[state_old][action] += alpha * (reward + self.gamma * np.max(self.Q[state_new]) - self.Q[state_old][action])

    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    def get_alpha(self, t):
        return max(self.min_alpha, min(1.0, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            current_state = self.discretize(self.env.reset())

            alpha = self.get_alpha(e)
            epsilon = self.get_epsilon(e)
            done = False
            i = 0

            x_out = np.zeros([4, 1000]); t_span = np.arange(0, 10, 0.01);
            for _ in range(1000):
                # self.env.render()
                action = self.choose_action(current_state, epsilon)
                obs, reward, done, _ = self.env.step(action)
                
                x_out[:,i] = obs
                
                new_state = self.discretize(obs)
                self.update_q(current_state, action, reward, new_state, alpha)
                current_state = new_state
                i += 1

            data = np.vstack([t_span, x_out]); data = data.T
            if(e % 100 == 0):
                drawCart(data,e)

            scores.append(i)
            mean_score = np.mean(scores)
            # if mean_score >= self.n_win_ticks and e >= 100:
            #     if not self.quiet: print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
            #     return e - 100
            if e % 100 == 0 and not self.quiet:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))

        if not self.quiet: print('Did not solve after {} episodes 😞'.format(e))
        return e

def drawCart(data, ep):
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
        min(data[:,1])*1.1,
        max(data[:,1])*1.1+.1,
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
        cart_plot.axis([-0.6,0.6,-.5,.5])
        cart_plot.plot([point[1]-.1,point[1]+.1],[0,0],'r-',lw=5)
        cart_plot.plot([point[1],point[1]+.4*sin(point[3])],[0,.4*cos(point[3])],'g-', lw=4)
    t = 0
    fps = 25.
    frame_number = 1
    for point in data:
        if point[0] >= t + 1./fps or not t:
            draw_point(point)
            t = point[0]
            fig.savefig("img/_tmp{0:03d}.png".format(frame_number))
            frame_number += 1

    print(os.system("ffmpeg -framerate 25 -i img/_tmp%03d.png  -c:v libx264 -r 30 -pix_fmt yuv420p out_ep{0:03d}.mp4".format(ep)))

if __name__ == "__main__":
    solver = QCartPoleSolver()
    solver.run()
# gym.upload('tmp/cartpole-1', api_key='')
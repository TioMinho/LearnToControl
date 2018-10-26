
import os
from math import sin, cos, pi
import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
env = env.unwrapped
# Policy gradient has high variance, seed for reproducability
env.seed(1)

## ENVIRONMENT Hyperparameters
state_size = 4
action_size = env.action_space.n

## TRAINING Hyperparameters
max_episodes = 1000
learning_rate = 0.01
gamma = 0.95 # Discount rate

def discount_and_normalize_rewards(episode_rewards):
	discounted_episode_rewards = np.zeros_like(episode_rewards)
	cumulative = 0.0
	for i in reversed(range(len(episode_rewards))):
		cumulative = cumulative * gamma + episode_rewards[i]
		discounted_episode_rewards[i] = cumulative

	mean = np.mean(discounted_episode_rewards)
	std = np.std(discounted_episode_rewards)
	discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

	return discounted_episode_rewards

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

with tf.name_scope("inputs"):
	input_ = tf.placeholder(tf.float32, [None, state_size], name="input_")
	actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
	discounted_episode_rewards_ = tf.placeholder(tf.float32, [None,], name="discounted_episode_rewards")

	# Add this placeholder for having this variable in tensorboard
	mean_reward_ = tf.placeholder(tf.float32 , name="mean_reward")

	with tf.name_scope("fc1"):
		fc1 = tf.contrib.layers.fully_connected(inputs = input_,
		                                        num_outputs = 10,
		                                        activation_fn=tf.nn.relu,
		                                        weights_initializer=tf.contrib.layers.xavier_initializer())

	with tf.name_scope("fc2"):
		fc2 = tf.contrib.layers.fully_connected(inputs = fc1,
		                                        num_outputs = action_size,
		                                        activation_fn= tf.nn.relu,
		                                        weights_initializer=tf.contrib.layers.xavier_initializer())

	with tf.name_scope("fc3"):
		fc3 = tf.contrib.layers.fully_connected(inputs = fc2,
		                                        num_outputs = action_size,
		                                        activation_fn= None,
		                                        weights_initializer=tf.contrib.layers.xavier_initializer())

	with tf.name_scope("softmax"):
		action_distribution = tf.nn.softmax(fc3)	

	with tf.name_scope("loss"):
		# tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
		# If you have single-class labels, where an object can only belong to one class, you might now consider using 
		# tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array. 
		neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fc3, labels = actions)
		loss = tf.reduce_mean(neg_log_prob * discounted_episode_rewards_) 

	with tf.name_scope("train"):
		train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Setup TensorBoard Writer
writer = tf.summary.FileWriter("tensorboard/pg/1")

## Losses
tf.summary.scalar("Loss", loss)

## Reward mean
tf.summary.scalar("Reward_mean", mean_reward_)

write_op = tf.summary.merge_all()

allRewards = []
total_rewards = 0
maximumRewardRecorded = 0
episode = 0
episode_states, episode_actions, episode_rewards = [],[],[]

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for episode in range(max_episodes+1):

		episode_rewards_sum = 0

		# Launch the game
		state = env.reset()

		# env.render()

		episode_states, episode_actions, episode_rewards = [],[],[]
		firstDone = False;
		x_out = np.zeros([4, 1000]); t_span = np.arange(0, 10, 0.01);
        # for _ in range(1000):
		for i in range(0, 1000):
			# Choose action a, remember WE'RE NOT IN A DETERMINISTIC ENVIRONMENT, WE'RE OUTPUT PROBABILITIES.
			action_probability_distribution = sess.run(action_distribution, feed_dict={input_: state.reshape([1,4])})

			action = np.random.choice(range(action_probability_distribution.shape[1]), p=action_probability_distribution.ravel())  # select action w.r.t the actions prob

			# Perform a
			new_state, reward, done, info = env.step(action)

			x_out[:,i] = new_state

			# Store s, a, r
			episode_states.append(state)

			# For actions because we output only one (the index) we need 2 (1 is for the action taken)
			# We need [0., 1.] (if we take right) not just the index
			action_ = np.zeros(action_size)
			action_[action] = 1

			episode_actions.append(action_)

			episode_rewards.append(reward)
			if done and not firstDone:
				firstDone = True
				# Calculate sum reward
				episode_rewards_sum = np.sum(episode_rewards)

				allRewards.append(episode_rewards_sum)

				total_rewards = np.sum(allRewards)

				# Mean reward
				mean_reward = np.divide(total_rewards, episode+1)


				maximumRewardRecorded = np.amax(allRewards)

				print("==========================================")
				print("Episode: ", episode)
				print("Reward: ", episode_rewards_sum)
				print("Mean Reward", mean_reward)
				print("Max reward so far: ", maximumRewardRecorded)

				# Calculate discounted reward
				discounted_episode_rewards = discount_and_normalize_rewards(episode_rewards)
				                
				# Feedforward, gradient and backpropagation
				loss_, _ = sess.run([loss, train_opt], feed_dict={input_: np.vstack(np.array(episode_states)),
				                                                 actions: np.vstack(np.array(episode_actions)),
				                                                 discounted_episode_rewards_: discounted_episode_rewards 
				                                                })


				                                                 
				# Write TF Summaries
				summary = sess.run(write_op, feed_dict={input_: np.vstack(np.array(episode_states)),
				                                                 actions: np.vstack(np.array(episode_actions)),
				                                                 discounted_episode_rewards_: discounted_episode_rewards,
				                                                    mean_reward_: mean_reward
				                                                })


				writer.add_summary(summary, episode)
				writer.flush()

				# Reset the transition stores
				episode_states, episode_actions, episode_rewards = [],[],[]

				# break

			state = new_state

		data = np.vstack([t_span, x_out]); data = data.T
		if(episode % 100 == 0):
			drawCart(data,episode)
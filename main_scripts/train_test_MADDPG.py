import numpy as np
import os, sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import matplotlib.pyplot as plt
import tensorflow as tf
from gym import spaces
from distutils.util import strtobool
main_proj_dir = os.path.dirname(sys.path[0])
sys.path.append(os.path.join(main_proj_dir, 'gym_multi_car_racing'))
sys.path.append(os.path.join(main_proj_dir, 'algorithms', 'MADDPG'))
sys.path.append(os.path.join(main_proj_dir, 'tools'))
import argparse
from aux_functions import *
from config_default_MADDPG import *
from MADDPG import *
from multi_car_racing import *

    
if __name__=="__main__":

    main_directory = main_proj_dir
    print('main directory to save the results: {}'.format(main_directory))
    def str2bool(v):
        return bool(strtobool(v)) 

    parser = argparse.ArgumentParser(description='Training the agent on single agent car racing')

    parser.add_argument('--num_cars', type=int, default=2, help='Number of cars to train the agent')
    parser.add_argument('--render', type=str2bool, nargs='?', const=True, default=True, help='Whether to display the game on the screen during training')
    parser.add_argument('--num_episodes', type=int, default=4000, help='Number of episodes to play the game')
    parser.add_argument('--max_episode_length', type=int, default=1000, help='Max number of timesteps for a single episode')
    parser.add_argument('--skip_frames', type=int, default=2, help='Number of frames to execute a single action taken by actor network')
    parser.add_argument('--max_neg_steps', type=int, default=200, help='Maximum number of consecutive negative steps w/ nonpositive reward to terminate the episode')
    parser.add_argument('--patience', type=int, default=1000, help='Patience for early stoping callback')

    parser.add_argument('--noise_type', type=str, default='ou', help='Type of noise to be added to the action (ou or normal)')
    parser.add_argument('--noise_std', type=str, default='0.0,0.2', help='Comma separated noise std for OU noise, check the length matches the action dimension, make sure it is close to 0 for testing')
    parser.add_argument('--noise_mean', type=str, default='0.0,-0.5', help='Comma separated noise mean, check the length matches the action dimension, make sure it is close to 0 for testing')
    parser.add_argument('--noise_scale', type=str, default='0.2,0.8', help='Noise scale for normal noise, make sure it is close to 0 for testing')
    parser.add_argument('--action_dim', type=int, default=2, help='Action dimension (2)')
    parser.add_argument('--tau', type=float, default=0.003, help='1 - polyak average weight to update target network weights')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for Q-value calculation')
    parser.add_argument('--max_memory_size', type=int, default=25000, help='Max size of replay buffer')
    parser.add_argument('--min_memory_size', type=int, default=2000, help='Min size of replay buffer')
    parser.add_argument('--action_divide_factor_steer', type=float, default=4, help='Divide the steering action generated to ease the training')
    parser.add_argument('--action_divide_factor_gb', type=float, default=6, help='Divide the gas/brake action generated to ease the training')
    
    parser.add_argument('--init_lr_actor', type=float, default=0.00001, help='Learning rate for actor model')
    parser.add_argument('--init_lr_critic', type=float, default=0.002, help='Learning rate for critic model')
    parser.add_argument('--decay_lr_steps', type=int, default=10, help='Exponential decay steps for lrs')
    parser.add_argument('--decay_lr_rate', type=float, default=0.99, help='Exponential decay rate for lrs')
    parser.add_argument('--decay_noise_steps', type=int, default=10, help='Exponential decay steps for noise')
    parser.add_argument('--decay_noise_rate', type=float, default=0.99, help='Exponential decay rate for noise')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training the networks')
    parser.add_argument('--max_to_keep', type=int, default=100, help='Maximum number of checkpoints to save (recommended to equal it to patience)')

    parser.add_argument('--train_or_test', type=str, default='test', help='Whether to train or load a model')
    parser.add_argument('--test_eps', type=int, default=1, help='Number of episodes to test the model')
    parser.add_argument('--ckpt_load', type=int, default=100, help='Number of checkpoint to load (put negative int not to load a checkpoint')
    parser.add_argument('--make_gif', type=str2bool, nargs='?', const=True, default=True, help='Whether to create gif during testing')

    # get the arguments
    args = parser.parse_args()
    num_cars = args.num_cars
    render = args.render
    num_episodes = args.num_episodes
    max_episode_length = args.max_episode_length
    skip_frames = args.skip_frames
    max_neg_steps = args.max_neg_steps

    noise_type = args.noise_type
    noise_std = np.array([float(i) for i in args.noise_std.split(',')])
    noise_mean =  np.array([float(i) for i in args.noise_mean.split(',')])
    noise_scale = np.array([float(i) for i in args.noise_scale.split(',')])
    action_dim = args.action_dim 
    tau = args.tau
    gamma = args.gamma
    max_memory_size = args.max_memory_size
    min_memory_size = args.min_memory_size
    action_divide_factor_steer = args.action_divide_factor_steer
    action_divide_factor_gb = args.action_divide_factor_gb
    batch_size = args.batch_size

    init_lr_actor = args.init_lr_actor
    init_lr_critic = args.init_lr_critic
    decay_lr_steps = args.decay_lr_steps
    decay_lr_rate = args.decay_lr_rate
    decay_noise_steps = args.decay_noise_steps
    decay_noise_rate =args.decay_noise_rate
    max_to_keep = args.max_to_keep

    train_or_test = args.train_or_test
    ckpt_load = args.ckpt_load
    test_eps = args.test_eps
    patience = args.patience

    make_gif = args.make_gif

    # create the config to overwrite to default one
    config_write = dict(num_cars=num_cars, noise_type=noise_type, noise_std=noise_std, noise_mean=noise_mean, noise_scale=noise_scale, action_dim=action_dim, tau=tau, gamma=gamma, max_memory_size=max_memory_size, 
                        init_lr_actor=init_lr_actor, init_lr_critic=init_lr_critic, 
                        decay_lr_steps=decay_lr_steps, decay_lr_rate=decay_lr_rate, decay_noise_steps=decay_noise_steps, decay_noise_rate=decay_noise_rate,
                        batch_size=batch_size, max_to_keep=max_to_keep, directory_to_save=main_directory,
                        train_or_test=train_or_test, ckpt_load=ckpt_load)

    # create the default config and overwrite
    config = set_config_default_MADDPG()
    config = add_settings_to_config(config, config_write)
    print('Settings used to start the training \n {}'.format(config))

    # create the game environment and reset it
    env = MultiCarRacing(num_agents=num_cars)
    env.reset()

    # create the learning agent
    STATE_H = 84   
    STATE_W = 96
    state_shape = (STATE_H, STATE_W, 3)
    agent = MADDPG(config, state_shape)
    
    if train_or_test == 'train':

        # create tensorflow writer
        agent.create_writer()
        callback = EarlyStoppingCallback(patience)

        # Fill the memory
        for st in range(min_memory_size):
            print('Filling memory, step: {}'.format(st))

            # render if requested
            if render:
                env.render()

            # get state and action
            state = env.render('state_pixels')
            action, action_before = agent.get_action(state)
            for i in range(num_cars):
                action[i][0] = action[i][0] / action_divide_factor_steer
                action[i][1:] = action[i][1:] / action_divide_factor_gb
            
            print('action: {}, action_before: {}'.format(action, action_before))

            # execute action on the environment
            reward = 0
            for i in range(skip_frames):
                next_state, r, terminal, _ = env.step(action)
                reward += r
            
            # if st < 50:
            #     continue

            # append the experience to the memory (append the action (generated by the actor + noise) )
            agent.memory.append_memory(state, action_before, reward, next_state, terminal)

            # reset the env if terminal state
            if terminal:
                env.reset()

        # average reward initialization
        avg_reward = tf.Variable([0]*num_cars, dtype=tf.float32)
        cum_reward = tf.Variable([0]*num_cars, dtype=tf.float32)

        # start running episodes
        start_ep = ckpt_load if ckpt_load > 0 else 0
        for ep_num in range(start_ep, num_episodes):

            # reset the environment, save models, update and reset noise, update learning rate
            env.reset()
            agent.save_models()
            agent.reset_update_noise(ep_num=ep_num)
            agent.update_network_lrs(ep_num=ep_num)

            # set episode variables to write tensorboard
            ep_reward = tf.Variable([0]*num_cars, dtype=tf.float32)
            ep_length = 0
            neg_reward_count = 0
            ep_done = False

            print('Episode {} starts...'.format(ep_num+1))
            while not ep_done:

                print('Episode {}, timestep {}/{}'.format(ep_num+1, ep_length, max_episode_length))

                # render the env if requested
                if render:
                    env.render()
                
                # get the action from last state 
                state = env.render('state_pixels')

                # get the action from actor (divide by 4 to train easier)
                action, action_before = agent.get_action(state)
                for i in range(num_cars):
                    action[i][0] = action[i][0] / action_divide_factor_steer
                    action[i][1:] = action[i][1:] / action_divide_factor_gb
                    
                # obtain the next state
                reward = [0]*num_cars
                for _ in range(skip_frames):
                    next_state, r, terminal, _ = env.step(action)
                    reward += r

                # if ep_length < 50:
                #     ep_length += 1
                #     continue
            
                print('Episode {}, step {}, action: {}, before clipping {}, episode reward: {}'.format(ep_num+1, ep_length, action, action_before, ep_reward.numpy()))
            
                # if sum of agents' reward is negative for max_neg_steps steps, episode ends
                if sum(reward) <= 0:
                    neg_reward_count += 1
                else:
                    neg_reward_count = 0

                # increase ep length and ep reward
                ep_length += 1
                ep_reward.assign_add(reward)

                if terminal or ep_length == max_episode_length or neg_reward_count == max_neg_steps:
                    ep_done = True
                    agent.memory.append_memory(state, action_before, 0., next_state, True)
                else:
                    agent.memory.append_memory(state, action_before, reward, next_state, terminal)

                # train the networks
                agent.train_step()

            # close the environment due to memory issues
            env.close()

            # get the avg and cumulative reward
            cum_reward.assign_add(ep_reward)
            print('Episode reward: {}'.format(ep_reward.numpy()))

            avg_reward = cum_reward.numpy() / (ep_num + 1)
            print('Average reward: {}'.format(avg_reward))

            # write avg and episode reward to tensorboard
            agent.def_summary(avg_reward, ep_reward, ep_num, 'train')
            
            # check the callback and if there is no improvement on the ep reward for the last 100 episodes, stop training
            callback_stop = callback(tf.reduce_sum(ep_reward))
            if callback_stop:
                break

    else:
        if make_gif:
            frames = []
        env.reset()

        if render:
            env.render()

        agent.create_writer()

        # average reward initialization
        avg_reward = tf.Variable([0]*num_cars, dtype=tf.float32)
        cum_reward = tf.Variable([0]*num_cars, dtype=tf.float32)

        for ep_num in range(test_eps):
            print('Test episodes: {}/{}'.format(ep_num+1, test_eps))
            ep_reward = tf.Variable([0]*num_cars, dtype=tf.float32)
            ep_done = False
            ep_length = 0

            while not ep_done:
                # get the video frame and append to frames
                if make_gif:
                    frames.append(env.render('rgb_array'))

                # render the env if requested
                if render:
                    env.render()

                # get the state and execute on the environment
                state = env.render('state_pixels')
                action, action_before = agent.get_action(state)
                for i in range(num_cars):
                    action[i][0] = action[i][0] / action_divide_factor_steer
                    action[i][1:] = action[i][1:] / action_divide_factor_gb

                reward = [0]*num_cars
                for _ in range(skip_frames):
                    next_state, r, terminal, _ = env.step(action)
                reward += r

                ep_length += 1
                ep_reward.assign_add(reward)

                if terminal or ep_length == max_episode_length:
                    ep_done = True

            # get the avg and cumulative reward
            cum_reward.assign_add(ep_reward)
            print('Episode reward: {}'.format(ep_reward.numpy()))

            avg_reward = cum_reward.numpy() / (ep_num + 1)
            print('Average reward: {}'.format(avg_reward))

            # write avg and episode reward to tensorboard
            agent.def_summary(avg_reward, ep_reward, ep_num, 'train')

            if ep_num != test_eps - 1: 
                print('Starting next_episode...')
            else:
                print('Creating the gif for the episode...')
        
        # create the gif out of the episode
        if make_gif:
            for car_id in range(num_cars):
                create_gif(frames, interval=100, dpi=80, save_path=os.path.join(main_directory, 'gifs', 'MADDPG_{}.gif'.format(car_id+1)), agent_id=car_id)
            

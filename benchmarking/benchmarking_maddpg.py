"""
This script is used to evaluate a trained agent and save test scores.
TODO for your agent:
    1. Implement your tester (see example in A3C/a3c.py)
    2. Import your tester agent below
    3. Fill in the name of your tester agent in get_agent_for_algo below
    4. To test your model run
        python benchmarking.py --algo [choose from 'a3c', 'dqn', 'ddpg', 'dddqn', 'ppo']
    5. Results of your run will be saved to ./results
Comments:
    - Currently number of testing episodes is set to 3 so that everybody can try it. We will increase this to ~1000
    - You can use the flag --render to show your agent while testing
    - You can set number of testing episodes via --num_test_episodes
"""
from model_tester_maddpg import *
from argparse import ArgumentParser
import os, sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'algorithms'))
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def get_agent_for_algo(algo):
    if algo == 'a3c':
        from A3C.a3c import A3CTesterAgent
        return A3CTesterAgent
    elif algo == 'ppo':
        from PPO.ppo_image import PPOTesterAgent
        return PPOTesterAgent
    elif algo == 'dqn':
        from DQN.DQNBenchmarkAgent import DQNTesterAgent
        return DQNTesterAgent
    elif algo == 'ddpg':
        from DDPG.DDPG import DDPGTesterAgent
        return DDPGTesterAgent 
    elif algo == 'ddqn':
        from dueling.DDQN import DDQNTesterAgent
        return DDQNTesterAgent
    elif algo == 'hc':
        from HC.hc import HCTesterAgent
        return HCTesterAgent
    elif algo == 'maddpg':
        from MADDPG.MADDPG import MADDPGTesterAgent
        return MADDPGTesterAgent

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--algo",
        default='maddpg',
        choices=("a3c", "dqn", "ddpg", "ddqn", "ppo", "hc", "maddpg"),
        type=str,
        help="Select algorithm.",
    )
    parser.add_argument(
        "--num_cars",
        default=2,
        type=int,
        help="Number of cars",
    )
    parser.add_argument(
        "--num_test_episodes",
        type=int,
        help="Number of episodes to run.",
        default=200
    )
    parser.add_argument(
        "--render",
        action='store_true',
        default=True,
        help="Render episodes during test.",
    )
    parser.add_argument(
        "--save_gif",
        action='store_true',
        default=False,
        help="Save gif of episode.",
    )
    parser.add_argument(
        "--save_gif_idx",
        type=int,
        help="Index of episode to save gif of (starting from 0).",
        default=0
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default='results',
        help='Folder to save log files',
    )

    args = parser.parse_args()

    agent = get_agent_for_algo(args.algo)()
    tester = ModelTester(agent=agent,
                         num_cars=args.num_cars,
                         num_test_episodes=args.num_test_episodes,
                         render=args.render,
                         save_gif=args.save_gif,
                         save_gif_idx=args.save_gif_idx,
                         )
    print('Testing {} algorithm...'.format(args.algo))
    eval_data = tester.evaluate()
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    log_file = os.path.join(args.output_folder, '{}.log'.format(args.algo))
    np_file = os.path.join(args.output_folder, '{}.npy'.format(args.algo))
    print('Testing finished, saving results to {} and {}...'.format(log_file, np_file))

    with open(log_file, 'w') as f:
        f.write('number of test episodes: {}\n'.format(eval_data['num_episodes']))
        f.write('average score: {}\n'.format(eval_data['avg_score']))
        f.write('std of score: {}\n'.format(eval_data['std_score']))
    np.save(np_file, eval_data)

    

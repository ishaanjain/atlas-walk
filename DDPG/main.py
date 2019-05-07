import argparse
from train import Train
from test import Test

parser = argparse.ArgumentParser()

# training args
parser.add_argument('--actor_learning_rate', type=float, default=1e-4)
parser.add_argument('--critic_learning_rate', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=64)  # changed from 100
parser.add_argument('--tau', type=float, default=1e-3)
parser.add_argument('--discount_rate', type=float, default=0.99)
parser.add_argument('--max_steps', type=int, default=200)
parser.add_argument('--max_episodes', type=int, default=10000)

# other args
parser.add_argument('--train', type=bool, default=True, help='Train or test network')
parser.add_argument('--render', type=bool, default=False, help='Render gym environment')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--checkpoint_frequency', type=int, default=100)
parser.add_argument('--display_frequency', type=int, default=100)

args = parser.parse_args()

if __name__ == '__main__':
    if args.train:
        train = Train(args)
        train.train()
    else:
        test = Test(args)
        test.test()

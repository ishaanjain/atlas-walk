import argparse
from train import Train
from test import Test

parser = argparse.ArgumentParser()

# training args
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--tau', type=float, default=1e-3)
parser.add_argument('--discount_rate', type=float, default=0.99)
parser.add_argument('--max_steps', type=int, default=200)
parser.add_argument('--update_repeat', type=int, default=10)
parser.add_argument('--max_episodes', type=int, default=5000)
parser.add_argument('--buffer_size', type=int, default=1000000)

# other args
parser.add_argument('--train', default=False, action='store_true', help='Train or test network')
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--render', default=False, action='store_true', help='Render gym environment')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--checkpoint_frequency', type=int, default=400)
parser.add_argument('--display_frequency', type=int, default=50)

args = parser.parse_args()

if __name__ == '__main__':
    if args.train:
        train = Train(args)
        train.train()
    elif args.test:
        test = Test(args)
        test.test()

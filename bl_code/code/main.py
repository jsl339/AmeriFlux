import argparse
from data_functions import load_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datadir", type=str, default=None,
                        help="where is the data saved?")
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_data(dir=args.datadir)
    print(data.head)


if __name__ == '__main__':
    main()

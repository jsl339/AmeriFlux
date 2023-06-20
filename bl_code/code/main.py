import argparse
from data_functions import load_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--userid", type=str, default=None,
                        help="AmeriFlux account user ID")
    parser.add_argument("-e", "--email", type=str, default=None,
                        help="AmeriFlux account email")    
    parser.add_argument("-d", "--datadir", type=str, default=None,
                        help="where is the data saved? (only necessary if not in ../data/)")
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_data(user_id=args.userid, user_email=args.email, dir=args.datadir)
    # print(data.head())

if __name__ == '__main__':
    main()

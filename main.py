import argparse
import os 

from evaluate import evaluate



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
       description='LPR')
    
    parser.add_argument('command',
                    metavar='<command>',
                    help="'train, detect, evaluate'")
   

    parser.add_argument('--device')

    args = parser.parse_args()

    if args.command == 'train':
        pass

    if args.command == 'evaluate':
        evaluate()
    
    if args.command == 'detect':
        pass
    
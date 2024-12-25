import argparse
from src import pipeline

parser = argparse.ArgumentParser()
parser.add_argument(
    '--InputPath', 
    help= 'Directory path containing files to be processed, or a single file path')

parser.add_argument(
    '--parser_name', 
    help= '')

parser.add_argument(
    '--chunking_strategy', 
    help= '')

parser.add_argument(
    '--retrieval_strategy', 
    help= '')

def main():
    args = parser.parse_args()

    pipeline.pipeline(args.InputPath, 
                      parser_name=args.parser_name, 
                      chunking_strategy=args.chunking_strategy, 
                      retrieval_strategy=args.retrieval_strategy)
    

if __name__ == '__main__':
    main()
import argparse
import annotator as ghex
import os


def main(args):
    print(args.dirpath)
    ghex.transcribe(dirpath=args.dirpath)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run annotation on a folder of waves')
    parser.add_argument(
        'dirpath', type=str, help='path to a folder of waves.'
    )
    parser.add_argument(
        'out_dirpath', type=str, help='path to a folder for outputing.'
    )

    main(parser.parse_args())
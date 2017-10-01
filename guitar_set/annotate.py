import argparse
import annotator as ghex
import os


def main(args):
    print(args.dirpath)
    jams_files = ghex.transcribe(dirpath=args.dirpath)
    return jams_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run annotation on a folder of waves')
    parser.add_argument(
        'dirpath', type=str, help='path to a folder of waves.'
    )

    main(parser.parse_args())
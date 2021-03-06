import annotator as ann
import argparse
import os

import guitar_set.interpreter
import guitar_set.scripts


def run_one(dirpath):
    man_jam = ann.csvs_to_jams(dirpath)
    man_jam.save(dirpath+'_man.jams')
    guitar_set.interpreter.sonify_jams(man_jam, dirpath + '_man.wav', q=0)


def run_many(dirpath):
    dir_list = guitar_set.scripts.get_immediate_subdirectories(dirpath)
    for dpath in dir_list:
        run_one(os.path.join(dirpath, dpath))


def main(args):
    if args.type == 1:
        run_one(args.dirpath)
    elif args.type == 2:
        run_many(args.dirpath)
    else:
        raise(AttributeError())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='manual annotation sonifier')
    parser.add_argument(
        'dirpath', type=str, help='path to the stem of interest')
    parser.add_argument(
        'type', type=int,
        help='1: run one, 2: run many'
    )

    main(parser.parse_args())

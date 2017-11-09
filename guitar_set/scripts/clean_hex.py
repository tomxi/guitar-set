import argparse
import glob
import os
import shutil
import tempfile

import sox

import mirapie.call_mira as mira
from guitar_set import util


# input_path = '/Users/tom/Music/DataSet/test-set/'
# out_path = '/Users/tom/Music/DataSet/test-set_cleaned/'


def run_one(input_path, output_dir=None):
    csv_path = 'guitar_set/resources/ghex_mira.csv'

    if output_dir is None:
        output_dir = input_path

    temp_path = tempfile.mkdtemp() + '/'
    # print(temp_path)

    output_mapping = {'0': {1: [1]},
                      '1': {1: [2]},
                      '2': {1: [3]},
                      '3': {1: [4]},
                      '4': {1: [5]},
                      '5': {1: [6]}
                      }

    for mix_type, remix_dict in sorted(output_mapping.items()):
        tfm = sox.Transformer()
        tfm.remix(remix_dictionary=remix_dict)
        output_path = os.path.join(temp_path, '{}.wav'.format(mix_type))
        tfm.build(input_path, output_path)

    mira.run(temp_path, csv_path)

    file_name = os.path.basename(input_path).split('.')[0] + '_cln.wav'

    cleaned_output_mapping = {
        file_name: {k: [v] for (k, v) in zip(range(1, 7), range(1, 7))}
    }

    cleaned_stems = [os.path.join(temp_path + 'Q/', f) for f in os.listdir(
        temp_path + 'Q/') if util.ext_f_condition(f, temp_path + 'Q/', 'wav')]
    cleaned_stems.sort()

    for f_name, remix_dict in sorted(cleaned_output_mapping.items()):
        cbn = sox.Combiner()
        cbn.remix(remix_dictionary=remix_dict)
        cbn.gain(normalize=True)
        out_path = os.path.join(output_dir, f_name)
        cbn.build(cleaned_stems, out_path, combine_type='merge')

    shutil.rmtree(temp_path)


def main(args):
    """clean the hex file"""
    base_dir = args.input_dir
    format_str = '/*hex.wav'
    input_paths = glob.glob(base_dir + format_str)
    todo_num = len(input_paths)
    for input_path in input_paths:
        print(input_path)
        print(todo_num)
        if os.path.exists(input_path.split('.')[0] + '_cln.wav'):
            pass  # output already exists!
        else:
            run_one(input_path, args.output_dir)
        todo_num -= 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Using Mira to clean a hex file')
    parser.add_argument(
        'input_dir', type=str, help='folder containing parser outputs.')
    # '/Users/tom/Music/DataSet/guitar_set/ed/'
    parser.add_argument(
        'output_dir', nargs='?', default=None, type=str,
        help='folder for the cleaned output. Default is same as input_dir.')
    main(parser.parse_args())

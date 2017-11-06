import os
import glob
import tempfile
import shutil
import sox
import mirapie.call_mira as mira
import argparse


# input_path = '/Users/tom/Music/DataSet/test_set/'
csv_path = 'guitar_set/resources/ghex_mira.csv'
# out_path = '/Users/tom/Music/DataSet/test_set_cleaned2/'


def run_one(input_path, csv_path, output_dir=None):

    if output_dir is None:
        output_dir = input_path

    temp_path = tempfile.mkdtemp() + '/'

    output_mapping = {'0': {1: [1]},
                      '1': {1: [2]},
                      '2': {1: [3]},
                      '3': {1: [4]},
                      '4': {1: [5]},
                      '5': {1: [6]}
                      }

    for mix_type, remix_dict in output_mapping.items():
        tfm = sox.Transformer()
        tfm.remix(remix_dictionary=remix_dict)
        output_path = os.path.join(temp_path, '{}.wav'.format(mix_type))
        tfm.build(input_path, output_path)

    mira.run(temp_path, csv_path)

    file_name = os.path.basename(input_path).split('.')[0] + 'c.wav'

    cleaned_output_mapping = {
        file_name: {k: [v] for (k, v) in zip(range(1, 7), range(1, 7))}
    }

    cleaned_stems = [os.path.join(temp_path, f) for f in os.listdir(temp_path)
                     if os.path.isfile(os.path.join(temp_path, f))]

    for file_name, remix_dict in cleaned_output_mapping.items():
        cbn = sox.Combiner()
        cbn.remix(remix_dictionary=remix_dict)
        out_path = os.path.join(output_dir,file_name)
        cbn.build(cleaned_stems, out_path, combine_type='merge')

    shutil.rmtree(temp_path)


def main(args):
    """clean the hex file"""
    base_dir = args.input_dir
    format_str = '/*hex.wav'
    input_paths = glob.glob(base_dir + format_str)
    print(len(input_paths))
    for input_path in input_paths:
        print(input_path)
        run_one(input_path, csv_path, args.output_dir)


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

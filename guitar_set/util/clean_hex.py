import os
import tempfile
import shutil
import sox
import mirapie.call_mira as mira
import argparse


# input_path = '/Users/tom/Music/DataSet/test_set/'
# csv_path = '/Users/tom/Music/DataSet/test_set/mira.csv'
# out_path = '/Users/tom/Music/DataSet/test_set_cleaned2/'


def run(input_path, csv_path, out_path):
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

    file_name = os.path.basename(input_path).split('.')[0] + '_cleaned.wav'
    cleaned_output_mapping = {
        file_name: {k: [v] for (k, v) in zip(range(1, 7), range(1, 7))}
    }
    cleaned_stems = [os.path.join(temp_path, f) for f in os.listdir(temp_path)
                     if os.path.isfile(os.path.join(temp_path, f))]

    for file_name, remix_dict in cleaned_output_mapping.items():
        cbn = sox.Combiner()
        cbn.remix(remix_dictionary=remix_dict)
        output_path = os.path.join(out_path, file_name)
        cbn.build(cleaned_stems, output_path, combine_type='merge')

    shutil.rmtree(temp_path)


def main(args):
    """clean the hex file"""
    input_paths = [os.path.join(args.input_dir, f) for f in os.listdir(
        args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))
        & (f.split('.')[1] == 'wav')]

    for f in input_paths:
        input_path = f
        run(input_path, args.csv_path, args.out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Using Mira to clean a hex file')
    parser.add_argument(
        'input_dir', type=str, help='path to the hex wav of interest')
    parser.add_argument(
        'csv_path', type=str,
        help='path to the interference csv'
    )
    parser.add_argument(
        'out_path', type=str,
        help='path to the output_folder'
    )

    main(parser.parse_args())

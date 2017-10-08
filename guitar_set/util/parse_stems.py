import argparse
import os
import sox
import csv


def process_session(input_dir, output_dir, pl_id):
    stem_names = [
        'comp_ref_1.wav',
        'solo_ref_1.wav',
        '0_c_1.wav',
        '1_c_1.wav',
        '2_c_1.wav',
        '3_c_1.wav',
        '4_c_1.wav',
        '5_c_1.wav',
        '0_s_1.wav',
        '1_s_1.wav',
        '2_s_1.wav',
        '3_s_1.wav',
        '4_s_1.wav',
        '5_s_1.wav',
        'comp_mix_1.wav',
        'solo_mix_1.wav'
    ]

    output_mapping = {
        'mic_comp': {1: [1]},
        'mic_solo': {1: [2]},
        'hex_comp': {k: [v] for (k, v) in zip(range(1,7), range(3,9))},
        'hex_solo': {k: [v] for (k, v) in zip(range(1,7), range(9,15))},
        'mix_comp': {1: [15, 16]},
        'mix_solo': {1: [17, 18]}
    }

    output_mapping_with_label = {
        'c0': {1: [3]},
        'c1': {1: [4]},
        'c2': {1: [5]},
        'c3': {1: [6]},
        'c4': {1: [7]},
        'c5': {1: [8]},
        's0': {1: [9]},
        's1': {1: [10]},
        's2': {1: [11]},
        's3': {1: [12]},
        's4': {1: [13]},
        's5': {1: [14]}
    }

    stem_paths = [os.path.join(input_dir, f) for f in stem_names]

    for stem_path in stem_paths:
        if not os.path.exists(stem_path):
            print('{} does not exist'.format(stem_path))
            return None

    sts, ets, labels = load_times()

    for st, et, label in zip(sts, ets, labels):
        print([st, et, label])

        # without extra folder
        for mix_type, remix_dict in output_mapping.items():
            cbn = sox.Combiner()
            cbn.trim(st, et)
            cbn.remix(remix_dictionary=remix_dict)
            output_path = os.path.join(
                output_dir, '{}_{}_{}.wav'.format(pl_id, label, mix_type))
            cbn.build(stem_paths, output_path, combine_type='merge')

        # with extra folder
        for mix_type, remix_dict in output_mapping_with_label.items():
            cbn = sox.Combiner()
            cbn.trim(st, et)
            cbn.remix(remix_dictionary=remix_dict)
            dir_name = '{}_{}_{}'.format(pl_id, label, mix_type[0])
            final_output_dir = os.path.join(output_dir, dir_name)
            if not os.path.exists(final_output_dir):
                os.makedirs(final_output_dir)
            output_path = os.path.join(
                final_output_dir, '{}.wav'.format(mix_type[1]))
            cbn.build(stem_paths, output_path, combine_type='merge')


def load_times():
    st_list = []
    et_list = []
    label_list = []

    with open('resources/cutting_times.csv', 'rb') as timing_csv:
        reader = csv.reader(timing_csv, delimiter=',')
        for row in reader:
            st_list.append(float(row[0]))
            et_list.append(float(row[1]))
            label_list.append(row[2])

    return st_list, et_list, label_list


def main(args):
    process_session(args.input_dir, args.output_dir, args.pl_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Logic Session Parser')
    parser.add_argument('input_dir', type=str, help='path to input folder')
    parser.add_argument('output_dir', type=str, help='path to output folder')
    parser.add_argument('pl_id', type=str, help='string to identify the '
                                                'player')
    main(parser.parse_args())

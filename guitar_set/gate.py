import os
import shutil
import tempfile

import sox

base_dir = '/Users/tom/Music/DataSet/test-set_cleaned/'
out_dir = '/Users/tom/Music/DataSet/test-set_gated'
todo_dir_list = [f for f in os.listdir(base_dir) if f.endswith(".wav")]

print(todo_dir_list)

for todo_hex in todo_dir_list:
    in_path = os.path.join(base_dir, todo_hex)
    out_path = os.path.join(out_dir, todo_hex.split('.')[0] + '_gated.wav')

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
        tfm_split = sox.Transformer()
        tfm_split.remix(remix_dictionary=remix_dict)
        temp_output_path = os.path.join(temp_path, '{}.wav'.format(mix_type))
        tfm_split.build(in_path, temp_output_path)

    out_temp_path = tempfile.mkdtemp() + '/'
    for f in os.listdir(temp_path):
        tfm = sox.Transformer()
        tfm.compand(attack_time=0.1, decay_time=0.5, soft_knee_db=None,
                    tf_points=[(-200, -200), (-30, -200),
                               (-29.9, -29.9), (0, 0)])
        temp_in_path = os.path.join(temp_path, f)
        out_path = os.path.join(out_temp_path, f)
        tfm.build(temp_in_path, out_path)

    file_name = os.path.basename(in_path).split('.')[0] + '_gated.wav'
    gated_stems = [os.path.join(out_temp_path, f) for f in os.listdir(
        out_temp_path)]
    gated_stems.sort()
    gated_output_mapping = {
        file_name: {k: [v] for (k, v) in zip(range(1, 7), range(1, 7))}
    }

    for f_name, remix_dict in sorted(gated_output_mapping.items()):
        cbn = sox.Combiner()
        cbn.remix(remix_dictionary=remix_dict)
        cbn.gain(normalize=True)
        out_path = os.path.join(out_dir, f_name)
        cbn.build(gated_stems, out_path, combine_type='merge')

    shutil.rmtree(temp_path)
    shutil.rmtree(out_temp_path)

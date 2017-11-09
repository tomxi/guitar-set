import os

import annotator as ann

base_dir = '/Users/tom/Music/DataSet/test-set_gated/'

todo_dir_list = [f for f in os.listdir(base_dir) if f.endswith(".wav")]

print(todo_dir_list)

todo_num = len(todo_dir_list)
for todo_hex in todo_dir_list:
    print([todo_hex, todo_num])
    out_path = os.path.join(base_dir, todo_hex.split('.')[0] + '.jams')
    # if os.path.exists(out_path):
    #     print('jams file already exist')
    # else:
    hex_path = os.path.join(base_dir, todo_hex)
    jam = ann.transcribe_hex(hex_path)
    print('saving jams to {}'.format(out_path))
    jam.save(out_path)
    todo_num -= 1

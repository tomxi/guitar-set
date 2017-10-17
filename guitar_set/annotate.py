import os
import annotator as ann

base_dir = '/Users/tom/Music/DataSet/test_set'

todo_dir_list = [f for f in os.listdir(base_dir) if f.endswith(".wav")]

print(todo_dir_list)

for todo_hex in todo_dir_list:
    print(todo_hex)
    hex_path = os.path.join(base_dir, todo_hex)
    jam = ann.transcribe_hex(hex_path)
    jam_path = os.path.join(base_dir, todo_hex.split('.')[0]+'.jams')
    print('saving jams to {}'.format(jam_path))
    jam.save(jam_path)


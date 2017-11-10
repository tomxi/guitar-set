import os
import json
import annotator as ann
import random
import string

base_dir = '/Users/tom/Music/DataSet/test-set_cleaned/'


N = 4
again = True
while again:
    random_dir = ''.join(
        random.choice(string.ascii_uppercase + string.digits) for _ in
        range(N))
    out_base_dir = os.path.join(base_dir, random_dir)
    try:
        os.mkdir(out_base_dir)
        again = False
    except OSError:
        continue


def random_param():
    # random param
    return None


#update pYin_param
with open('resources/pYin_param.json', 'rw') as p:
    param = random_param()
    json.dump(param, p)


todo_dir_list = [f for f in os.listdir(base_dir) if f.endswith(".wav")]

print(todo_dir_list)

todo_num = len(todo_dir_list)
for todo_hex in todo_dir_list:
    print([todo_hex, todo_num, param])
    out_path = os.path.join(out_base_dir, todo_hex.split('.')[0] + '.jams')
    hex_path = os.path.join(base_dir, todo_hex)
    jam = ann.transcribe_hex(hex_path)
    print('saving jams to {}'.format(out_path))
    jam.save(out_path)
    todo_num -= 1


with open(out_base_dir+'param.json', 'w') as stream:
    json.dump(param, stream)


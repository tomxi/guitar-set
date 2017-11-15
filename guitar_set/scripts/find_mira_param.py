import os
import random
import string
import yaml
import json

import clean_hex as cln

base_dir = '/Users/tom/Music/DataSet/test-set/'
out_dir = '/Users/tom/Music/DataSet/test-set_mira_search1/'

def random_param():
    param = {"minimal_interference": random.uniform(0,0.9),
             "n_iter": random.choice([3,4,5,6,7,8]),
             "nfft": 2 ** random.choice([10,11,12,13,14,15,16]),
             "overlap": random.uniform(0.5,0.95)
             }

    return param


while True:
    N = 4
    again = True
    while again:
        random_dir = ''.join(
            [random.choice(string.ascii_uppercase + string.digits) for _ in
            range(N)]
        )
        out_base_dir = os.path.join(out_dir, random_dir)
        try:
            os.mkdir(out_base_dir)
            again = False
        except OSError:
            continue

    # update preset_batch.yml
    with open("mirapie/preset_batch.yml", 'rb') as stream:
        try:
            preset_yml = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    param = random_param()
    for k,v in param.items():
        preset_yml[1][k] = v

    with open("mirapie/preset_batch.yml", "wb") as f:
        yaml.dump(preset_yml, f, encoding='utf-8')

    todo_dir_list = [f for f in os.listdir(base_dir) if f.endswith(".wav")]

    print(param, todo_dir_list)

    todo_num = len(todo_dir_list)
    for todo_hex in todo_dir_list:
        print(todo_hex, random_dir, todo_num)
        # out_path = os.path.join(out_base_dir, todo_hex.split('.')[0] + '.jams')
        hex_path = os.path.join(base_dir, todo_hex)

        cln.run_one(hex_path,out_base_dir)

        todo_num -= 1

    with open(os.path.join(out_base_dir,'param.json'), 'w') as stream:
        json.dump(param, stream)


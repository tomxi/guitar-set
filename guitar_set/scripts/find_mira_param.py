import os
import random
import string
import yaml
import json

import guitar_set.annotate as ann
import clean_hex as cln

base_dir = '/Users/tom/Music/DataSet/test-set/'
out_dir = '/Users/tom/Music/DataSet/test-set_mira_search2/'

def random_param():
    param = {"minimal_interference": random.uniform(0,0.4),
             "n_iter": random.choice([3,4,5]),
             "nfft": 2 ** random.choice([12,13,14,15]),
             "overlap": random.uniform(0.8,0.95)
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
    print('update preset_batch.yml')
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
        hex_path = os.path.join(base_dir, todo_hex)
        cln.run_one(hex_path,out_base_dir)
        todo_num -= 1

    ann.do(out_base_dir)

    with open(os.path.join(out_base_dir,'param.json'), 'w') as stream:
        json.dump(param, stream)


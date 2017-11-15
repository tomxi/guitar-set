import os
import json
import guitar_set.annotator as ann
import random
import string

base_dir = '/Users/tom/Music/DataSet/test-set_cleaned/'
out_dir = '/Users/tom/Music/DataSet/test-set_cleaned_output2/'

def random_param():
    param = {"threshdistr": 2,
             "lowampsuppression": random.uniform(0.075, 0.1),
             "outputunvoiced": 0,
             "precisetime": 0,
             "prunethresh": random.uniform(0.01, 0.1),
             "onsetsensitivity": random.uniform(0.75, 0.85)}

    return param


while True:
    # generate random folder of N char in len
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

    # update pYin_param
    with open('resources/pYin_param.json', 'w') as p:
        param = random_param()
        json.dump(param, p)

    todo_dir_list = [f for f in os.listdir(base_dir) if f.endswith(".wav")]

    print(param, todo_dir_list)

    # do all in todo_dir_list
    todo_num = len(todo_dir_list)
    for todo_hex in todo_dir_list:
        print(todo_hex, random_dir, todo_num)
        out_path = os.path.join(out_base_dir, todo_hex.split('.')[0] + '.jams')
        hex_path = os.path.join(base_dir, todo_hex)
        jam = ann.transcribe_hex(hex_path)
        print('saving jams to {}'.format(out_path))
        jam.save(out_path)
        todo_num -= 1

    # save param as param.json
    with open(os.path.join(out_base_dir,'param.json'), 'w') as stream:
        json.dump(param, stream)


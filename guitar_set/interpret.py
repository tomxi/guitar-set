import os

import jams

from guitar_set import interpreter as itp
from guitar_set import util

base_dir = '/Users/tom/Music/DataSet/guitar-set_cln/'
out_dir = '/Users/tom/Music/DataSet/guitar-set_cln/itp/'
jams_list = util.get_all_ext(base_dir, ext='.jams')

for jam_path in jams_list:
    print(jam_path)
    jam = jams.load(jam_path)
    out_path = os.path.join(
        out_dir, os.path.split(jam_path)[1].split('.')[0])
    itp.sonify_jams(jam, out_path + '_syn.wav', q=0)
    itp.visualize_jams(jam, save_path=out_path + '_note.png')
    itp.tablaturize_jams(jam, save_path=out_path + '_fret.png')

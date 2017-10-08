import os
import annotator as ann

base_dir = '/Users/tom/Music/DataSet/test_set_debleed'

todo_dir_list = ann.get_immediate_subdirectories(base_dir)

print(todo_dir_list)

for todo_dir in todo_dir_list:
    print(todo_dir)
    current_task = todo_dir
    dirpath = os.path.join(base_dir, current_task)
    ann.transcribe(dirpath)
    # combine jams files to make midi
    jam = ann.jamses_to_jams(dirpath)
    jam.save(dirpath+'.jams')
    # man_jam = ann.csvs_to_jams(dirpath)
    # man_jam.save(dirpath+'_man.jams')
    midi_file = ann.jams_to_midi(jam)
    ann.sonify(midi_file, dirpath + '_trans.wav')
    # man_midi = ann.jams_to_midi(man_jam)
    # ann.sonify(man_midi, dirpath + '_manual.wav')

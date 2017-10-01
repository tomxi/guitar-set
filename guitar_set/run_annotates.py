import os
import annotator as ann

base_dir = '/Users/tom/Documents/REPO/mirapie/results'


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

todo_dir_list = get_immediate_subdirectories(base_dir)

while len(todo_dir_list) is not 0:
    print('todo length:{}'.format(len(todo_dir_list)))
    current_task = todo_dir_list[-1] # because using List.pop() later.
    dirpath = os.path.join(base_dir, current_task)
    jams_files = ann.transcribe(dirpath)
    # combine jams files to make midi
    midi_file = ann.jams_to_midi(jams_files)
    ann.sonify(midi_file, dirpath+'.wav')
    todo_dir_list.pop()


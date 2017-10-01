import os

base_dir = '/Users/tom/Documents/REPO/mirapie/results'


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

todo_dir_list = get_immediate_subdirectories(base_dir)

while len(todo_dir_list) is not 0:
    print('todo length:{}'.format(len(todo_dir_list)))
    current_task = todo_dir_list[-1] # because using List.pop() later.
    dirpath = os.path.join(base_dir, current_task)
    out_dirpath = base_dir
    command = 'python annotate.py {} {}'.format(dirpath, out_dirpath)
    err = os.system(command)
    if err:
        print('errored, continuing')
        continue
    else: # successful
        todo_dir_list.pop()


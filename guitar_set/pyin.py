import os




def mono_anal(stem_path, open_string_midi):
    """save jams with same stem_path"""
    done = False
    cmd = 'python guitar_set/scripts/mono_anal_script.py {} {}'.format(
        stem_path, open_string_midi)
    while not done:
        err = os.system(cmd)
        if err:
            print('vamp.collect errored, trying again...')
        else: # successful, no seg fault
            done = True

    return 0
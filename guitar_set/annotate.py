import os
import jams
import guitar_set.annotator as ann

pYin_param = {"threshdistr": 2,
         "lowampsuppression": 0.08,
         "outputunvoiced": 0,
         "precisetime": 0,
         "prunethresh": 0.05,
         "onsetsensitivity": 0.8}

# with open('../resources/pYin_param.json', 'rw') as p:
#     json.dump(pYin_param, p)

def do(base_dir):
    # transcribe everything in the base_dir
    todo_dir_list = [f for f in os.listdir(base_dir) if f.endswith(".wav")]

    print(todo_dir_list)

    todo_num = len(todo_dir_list)
    for todo_hex in todo_dir_list:
        print([todo_hex, todo_num])
        out_path = os.path.join(base_dir, todo_hex.split('.')[0] + '.jams')
        if os.path.isfile(out_path):
            todo_num -= 1
            continue
        hex_path = os.path.join(base_dir, todo_hex)
        jam = ann.transcribe_hex(hex_path)
        print('saving jams to {}'.format(out_path))
        jam.save(out_path)
        todo_num -= 1


def dir_to_score(ref_dir, est_dir):
    # ref_dir and est_dir need to contain all 8 jams files of the test-set
    jams_list = [f for f in os.listdir(est_dir) if len(f.split('.')) > 1 and f.split('.')[1] == 'jams']
    ref_list = [f for f in os.listdir(ref_dir) if len(f.split('.')) > 1 and f.split('.')[1] == 'jams']
    jams_list.sort()
    ref_list.sort()

    # combine and collect all annotations in big_est and big_ref
    big_est = jams.Annotation('pitch_midi')
    big_ref = jams.Annotation('pitch_midi')
    big_est.duration = 0
    big_ref.duration = 0
    for e, r in zip(jams_list, ref_list):
        est_jams = jams.load(os.path.join(est_dir, e))
        ref_jams = jams.load(os.path.join(ref_dir, r))
        # print(e,r)
        for i in range(6):
            try:
                est_ann = est_jams.search(namespace='note_midi')[i]
            except IndexError:
                est_ann = est_jams.search(namespace='pitch_midi')[i]
                
            try:
                ref_ann = ref_jams.search(namespace='note_midi')[i]
            except IndexError:
                ref_ann = ref_jams.search(namespace='pitch_midi')[i]
            
            t_offset = i * est_ann.duration + big_est.duration
            big_est.duration += est_ann.duration
            big_ref.duration += ref_ann.duration
            for obs in est_ann:
                big_est.append(time=obs.time + t_offset, duration=obs.duration,
                               value=obs.value)
            for obs in ref_ann:
                big_ref.append(time=obs.time + t_offset, duration=obs.duration,
                               value=obs.value)

    scores = jams.eval.transcription(big_ref, big_est)
    return scores


def get_dir_list(dir_path):
    dir_list = [os.path.join(dir_path, d) for d in os.listdir(dir_path)
                if os.path.isdir(os.path.join(dir_path, d))]
    return dir_list


if __name__ == '__main__':
    base_dir = '/Users/tom/Music/DataSet/test-set_processed/'
    do(base_dir)
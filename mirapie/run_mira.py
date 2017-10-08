import os
import yaml
import argparse

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


# in_dir = '/Users/tom/Desktop/test_set/'
# cvs_path = 'resources/ghex_mira.csv'


def run(in_dir, cvs_path):
    dir_list = get_immediate_subdirectories(in_dir)

    for my_dir in dir_list:

        print(my_dir)
        preset_yml = 1
        with open("preset_batch.yml", 'rb') as stream:
            try:
                preset_yml = yaml.load(stream)
                print(preset_yml)
                preset_yml[1]['preset_name'] = my_dir
            except yaml.YAMLError as exc:
                print(exc)

        with open("preset.yml", "wb") as f:
            yaml.dump(preset_yml, f, encoding='utf-8')

        final_dir = os.path.join(in_dir, my_dir)
        os.system('./mirapie.py {} {} -p 1 -m 1'.format(
            final_dir, cvs_path))


def main(args):
    run(in_dir=args.input_dir, cvs_path=args.cvs_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mira batch runner')
    parser.add_argument('input_dir', type=str, help='path to input folder')
    parser.add_argument('cvs_path', type=str, help='path to cvs')

    main(parser.parse_args())

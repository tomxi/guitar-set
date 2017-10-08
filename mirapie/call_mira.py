import os
import yaml
import argparse


def run(input_path, csv_path):
    preset_yml = 1
    with open("mirapie/preset_single.yml", 'rb') as stream:
        try:
            preset_yml = yaml.load(stream)
            preset_yml[1]['output_folder_path'] = input_path
        except yaml.YAMLError as exc:
            print(exc)

    with open("preset.yml", "wb") as f:
        yaml.dump(preset_yml, f, encoding='utf-8')

    os.system('./mirapie/mirapie.py {} {} -p 1 -m 1'.format(
        input_path, csv_path))


def main(args):
    run(input_path=args.input_path, csv_path=args.csv_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mira batch runner')
    parser.add_argument('input_path', type=str, help='path to input folder')
    parser.add_argument('csv_path', type=str, help='path to cvs')
    main(parser.parse_args())
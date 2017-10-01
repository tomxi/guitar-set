import argparse
import annotator as ghex
import os


def main(args):
    print(args.dirpath)
    output, dur = ghex.transcribe(dirpath=args.dirpath)
    midi_out = ghex.output_to_midi(output)
    output_path = os.path.join(args.out_dirpath, args.dirpath + '.wav')
    print('about to sonify')
    ghex.sonify(midi_out, output_path)
    print('sonify done, saved as {}'.format(output_path))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run annotation on a folder of waves')
    parser.add_argument(
        'dirpath', type=str, help='path to a folder of waves.'
    )
    parser.add_argument(
        'out_dirpath', type=str, help='path to a folder for outputing.'
    )

    main(parser.parse_args())
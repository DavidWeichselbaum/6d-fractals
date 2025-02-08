import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="6D Fractal Viewer", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--load", type=str, metavar="PATH", help="Path to save file.", default="./saves/mandelbrot.yaml"
    )
    return parser.parse_args()

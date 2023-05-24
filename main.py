from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model_name")
parser.add_argument("--pose")
args = parser.parse_args()


def main(args):
    print(f"model_name: {args.model_name}, pose: {args.pose}")


if __name__ == "__main__":
    main(args)

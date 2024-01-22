from argparse import ArgumentParser
from subprocess import run, PIPE

GPIO_1 = "17"


@DeprecationWarning
def main():
    parser = ArgumentParser()
    parser.add_argument("action", choices=["start", "stop", "status"], help="Start, stop or show status of recording")
    args = parser.parse_args()

    if args.action == "start":
        run(['pigs', 'w', GPIO_1, '1'])

    if args.action == "stop":
        run(['pigs', 'w', GPIO_1, '0'])

    if args.action == "status":
        status = run(["pigs", "r", GPIO_1], stdout=PIPE)
        print(f"Recoding ist currently {'started' if '1' in status.stdout.decode('UTF-8') else 'stopped'}!")


if __name__ == '__main__':
    main()

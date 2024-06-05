from rpi_hardware_pwm import HardwarePWM
from argparse import ArgumentParser
import sys


def main():
    parser = ArgumentParser()
    parser.add_argument("action", choices=["start", "stop"], help="Start or stop the sync signal")
    parser.add_argument("--frequency", "-f", help="Frequency of sync signal", type=float, required='start' in sys.argv)
    parser.add_argument("--length", "-l", help="Length of 'on' portion of signal in micro seconds", type=int,
                        default=200)
    parser.add_argument("--channel", "-c", help="Used PWM channel. Can be 0, 1 or 2 for both", choices=[0, 1, 2],
                        default=0, type=int)
    args = parser.parse_args()

    if args.action == 'start':
        start_sync_signal(args.frequency, args.length, args.channel)
    elif args.action == 'stop':
        stop_sync_signal(args.channel)


def calc_duty_cycle(frequency, duty_length=200):
    """
    calculates duty cycle for pwm signal for given frequency and length of on segment in micro seconds
    :param frequency: frequency of pwm signal
    :param duty_length: length of on signal in micro seconds (default 200 us)
    :return: percentage value of duty cycle
    """
    return duty_length * frequency * 1e-4


def start_sync_signal(frequency: float, duty_length=200, pwm_channel=0):
    duty_cycle = calc_duty_cycle(frequency, duty_length)
    if pwm_channel == 2:
        pwm_0 = HardwarePWM(pwm_channel=0, hz=frequency)
        pwm_1 = HardwarePWM(pwm_channel=1, hz=frequency)
        pwm_0.start(duty_cycle)
        pwm_1.start(duty_cycle)
    else:
        pwm = HardwarePWM(pwm_channel=pwm_channel, hz=frequency)
        pwm.start(duty_cycle)


def stop_sync_signal(pwm_channel=0):
    if pwm_channel == 2:
        HardwarePWM(0, 60).stop()
        HardwarePWM(1, 60).stop()
    else:
        HardwarePWM(pwm_channel, 60).stop()


if __name__ == '__main__':
    main()

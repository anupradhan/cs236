import os
import time


def main():
    while True:
        print("Pausing for 4 hours.")
        time.sleep(60*60*4)
        print("Finished")


if __name__ == "__main__":
    main()
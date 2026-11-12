from pathlib import Path


def main():
    Path("first-test.txt").write_text("Hello World")


if __name__ == "__main__":
    main()

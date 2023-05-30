from university import Universities


def main():
    universities = Universities.from_schools("test.txt")
    universities.to_file("university/dataset.csv")
    print(universities)


if __name__ == "__main__":
    main()

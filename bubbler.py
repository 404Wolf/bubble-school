from university import Universities
import matplotlib.pyplot as plt


def main():
    universities = Universities.from_file("university/dataset.csv")

    clustered, centroids = universities.cluster(
        2, ("graduates", "endowment"), resolution=1
    )
    print(clustered["cluster"])

    plt.scatter(
        x=clustered["graduates"],
        y=clustered["endowment"],
        c=clustered["cluster"],
    )
    plt.scatter(
        x=centroids["graduates"],
        y=centroids["endowment"],
        c="red",
    )
    plt.show()


if __name__ == "__main__":
    main()

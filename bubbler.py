from university import Universities
import matplotlib.pyplot as plt


def main():
    universities = Universities.from_file("university/dataset.csv")

    clustered, centroids = universities.cluster(
        3, ("endowment", "cost"), resolution=150
    )

    plt.scatter(
        x=clustered["cost"],
        y=clustered["endowment"],
        c=clustered["cluster"],
    )
    plt.scatter(
        x=centroids["cost"],
        y=centroids["endowment"],
        c="red",
    )
    plt.show()


if __name__ == "__main__":
    main()

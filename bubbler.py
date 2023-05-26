from university import Universities
import matplotlib.pyplot as plt


def main():
    universities = Universities.from_file("university/dataset.csv")

    clustered = universities.cluster(4, ("endowment", "cost"), resolution=300)
    # while (uniques := clustered['cluster'].unique()[0]) != 3:
    #     print(uniques)
    #     clustered, centroids = universities.cluster(4, ("endowment", "cost"), resolution=2)
    #
    clustered.plot(
        x="cost",
        y="endowment",
        kind="scatter",
        c=clustered["cluster"],
    )
    # centroids.plot(x="cost", y="endowment", c="red", kind="scatter")
    plt.show()


if __name__ == "__main__":
    main()

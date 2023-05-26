import numpy as np

from university import Universities
import matplotlib.pyplot as plt


def main():
    universities = Universities.from_file("university/dataset.csv")

    clustered = universities.cluster(
        3, ("endowment", "graduates", "cost"), resolution=40
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    colors = clustered["cluster"].values
    centroids = clustered["centroid"].values
    for centroid_index, centroid in enumerate(clustered.index):
        if centroids[centroid_index]:
            colors[centroid_index] = -1

    ax.scatter(
        xs=clustered["endowment"],
        ys=clustered["graduates"],
        zs=clustered["cost"],
        c=colors
    )
    ax.set_xlabel("Endowment")
    ax.set_ylabel("Graduates")
    ax.set_zlabel("Cost")
    ax.set_title("Clustered Universities")
    plt.show()


if __name__ == "__main__":
    main()

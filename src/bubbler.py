from university import Universities
import matplotlib.pyplot as plt


def two_dimensions_test():
    universities = Universities.from_file("university/dataset.csv")

    clustered, centroids = universities.cluster(
        4, ("graduates", "endowment"), resolution=120
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


def three_dimensions_test():
    universities = Universities.from_file("university/dataset.csv")

    clustered, centroids = universities.cluster(
        3, ("cost", "endowment", "facultySalary"), resolution=3
    )
    for cluster in clustered.groupby("cluster"):
        print(cluster)
        print("\n\n")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    colors = clustered["cluster"].values
    ax.scatter(
        xs=clustered["cost"],
        ys=clustered["endowment"],
        zs=clustered["facultySalary"],
        c=colors,
    )
    ax.scatter(
        xs=centroids[:, 0],
        ys=centroids[:, 1],
        zs=centroids[:, 2],
        c="red",
    )
    ax.set_xlabel("Cost")
    ax.set_ylabel("Endowment")
    ax.set_zlabel("Faculty Salary")
    ax.set_title("Clustered Universities")
    plt.show()


if __name__ == "__main__":
    three_dimensions_test()

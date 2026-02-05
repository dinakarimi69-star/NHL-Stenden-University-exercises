import csv
import numpy as np
import matplotlib.pyplot as plt

def create_csv():
    f = open("data_file.csv", "w")
    w = csv.writer(f)
    _ = w.writerow(["precision", "recall"])
    w.writerows([
        [0.013, 0.951],
        [0.376, 0.851],
        [0.441, 0.839],
        [0.570, 0.758],
        [0.635, 0.674],
        [0.721, 0.604],
        [0.837, 0.531],
        [0.860, 0.453],
        [0.962, 0.348],
        [0.982, 0.273],
        [1.0, 0.0]
    ])
    f.close()
    print("CSV File Created: 'data_file.csv'")

def plot_data(csv_file_path: str):
    precision = []
    recall = []

    # Read data from CSV file
    with open(csv_file_path) as result_csv:
        reader = csv.DictReader(result_csv)
        for row in reader:
            precision.append(float(row["precision"]))
            recall.append(float(row["recall"]))

    plt.plot(precision, recall)
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.grid(True)

    # Save plot as PNG file 
    plt.savefig("precision_recall_plot.png")
    print("Precision-Recall plot saved as 'precision_recall_plot.png'")

create_csv()  # Create the CSV file
plot_data("data_file.csv")  # Plot data from the CSV file

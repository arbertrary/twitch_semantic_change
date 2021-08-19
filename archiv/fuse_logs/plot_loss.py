import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    files = os.listdir(".")
    fig = plt.figure()
    plt.title("AutoFusion loss over 50 epochs for 12 synthetic months")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    fig.set_size_inches(10, 5.4)

    for file in files:
        if file.startswith("logs"):
            with open(file, "r") as logfile:
                losses = []
                for line in logfile.readlines():
                    if line.startswith("0.0"):
                        losses.append(float(line.strip()))

                plt.plot(losses)
    plt.savefig("/home/armin/masterarbeit/thesis/ausarbeitung/pics/figures/fuse_loss.png")

    plt.show()

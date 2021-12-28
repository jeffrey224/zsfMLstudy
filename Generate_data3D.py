import numpy as np
import matplotlib.pyplot as plt

def f(x,y):
    return np.heaviside(x, 0.5) + x * np.sin(3 * x) + y * np.sin(3 * y)


def main():
    ntrain = 64
    ntest = 32

    x = np.linspace(-(3 ** 0.5), 3 ** 0.5, num=ntrain)
    y = np.linspace(-(3 ** 0.5), 3 ** 0.5, num=ntrain)
    X,Y = np.meshgrid(x,y)
    Z = f(X,Y)
    np.savetxt("train3d.txt", np.vstack((X.flat, Y.flat, Z.flat)).T)

    x = np.linspace(-(3 ** 0.5), 3 ** 0.5, num=ntest)
    y = np.linspace(-(3 ** 0.5), 3 ** 0.5, num=ntest)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    np.savetxt("test3d.txt", np.vstack((X.flat, Y.flat, Z.flat)).T)

    train_data = np.loadtxt("train3d.txt").astype(np.float32)
    print(train_data[:,2:])
  #  train_x, train_y = train_data[:, :1], train_data[:, 1:]


    ax = plt.axes(projection='3d')
    ax.plot_wireframe(X,Y,Z, color='red')
    plt.show()

if __name__ == "__main__":
    main()
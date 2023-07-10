import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Exercise 9
def gibbs_sampler_method1(Temp, size, sweeps):
    img_size = size + 2
    image = np.random.randint(low=0, high=2, size=(img_size, img_size)) * 2 - 1
    image[:1, :] = 0
    image[-1:, :] = 0
    image[:, :1] = 0
    image[:, -1:] = 0
    for s in range(sweeps):
        for i in range(1, img_size-1):
            for j in range(1, img_size - 1):
                neighbors_sum = (image[i-1][j] + image[i+1][j] + image[i][j-1] + image[i][j+1]) /Temp
                teta1 = np.exp(neighbors_sum)
                teta2 = np.exp(-neighbors_sum)
                curr_p = np.array([teta1, teta2]) / (teta1 + teta2)
                image[i][j] =np.random.choice([1,-1], p=curr_p)

    return image[1:-1, 1:-1]


def expectation_method_1(size):
    iterations = 10000
    temps = [1, 1.5, 2]
    Es_12 = [0 for temp in temps]
    Es_18 = [0 for temp in temps]
    for i in range(len(temps)):
        for j in range(iterations):
            print(j)
            image = gibbs_sampler_method1(temps[i], size, 25)
            Es_12[i] += image[0][0] * image[1][1]
            Es_18[i] += image[0][0] * image[7][7]
        Es_12[i] /= iterations
        Es_18[i] /= iterations

    return Es_12, Es_18



def gibbs_sampler_method2(Temp, size, sweeps):
    img_size = size + 2
    image = np.random.randint(low=0, high=2, size=(img_size, img_size)) * 2 - 1
    image[:1, :] = 0
    image[-1:, :] = 0
    image[:, :1] = 0
    image[:, -1:] = 0
    e_12 = 0
    e_17 = 0
    for s in range(sweeps):
        print(s)
        for i in range(1, img_size-1):
            for j in range(1, img_size - 1):
                neighbors_sum = (image[i-1][j] + image[i+1][j] + image[i][j-1] + image[i][j+1] ) /Temp
                teta1 = np.exp(neighbors_sum)
                teta2 = np.exp(-neighbors_sum)
                curr_p = np.array([teta1, teta2]) / (teta1 + teta2)
                image[i][j] =np.random.choice([1,-1], p=curr_p)
        if s >= 100:
            e_12 += image[1][1] * image[2][2]
            e_17 += image[1][1] * image[size-2][size-2]
    e_12 /= (sweeps - 100)
    e_17 /= (sweeps - 100)
    return e_12, e_17

def expectation_method_2(size):
    sweeps = 25000
    temps = [1, 1.5, 2]
    Es = [(gibbs_sampler_method2(temp, size, sweeps)) for temp in temps]
    return Es



# Exercise 10
# Sampling from posterior distribution
def gibbs_sampler_ex10_3(Temp, size, sweeps , y):
    # pad the matrix with zeros
    y = np.pad(y, 1, mode='constant', constant_values=0)
    img_size = size + 2
    image = np.random.randint(low=0, high=2, size=(img_size, img_size)) * 2 - 1
    image[:1, :] = 0
    image[-1:, :] = 0
    image[:, :1] = 0
    image[:, -1:] = 0
    for s in range(sweeps):
        print(s)
        for i in range(1, img_size-1):
            for j in range(1, img_size - 1):
                neighbors_sum = (image[i - 1][j] + image[i + 1][j] + image[i][j - 1] + image[i][j + 1]) / Temp
                teta1 = np.exp(neighbors_sum - (y[i][j] - 1) ** 2 / 8)
                teta2 = np.exp(-neighbors_sum - (y[i][j] + 1) ** 2 / 8)
                curr_p = np.array([teta1, teta2]) / (teta1 + teta2)
                image[i][j] =np.random.choice([1,-1], p=curr_p)

    return image[1:-1, 1:-1]

# ICM restoration
def gibbs_sampler_ex10_4(Temp, size, y):
    # pad the matrix with zeros
    y = np.pad(y, 1, mode='constant', constant_values=0)
    img_size = size + 2
    image = np.random.randint(low=0, high=2, size=(img_size, img_size)) * 2 - 1
    image[:1, :] = 0
    image[-1:, :] = 0
    image[:, :1] = 0
    image[:, -1:] = 0
    changes = 21
    s = 0
    while changes > 20:
        print(s)
        s += 1
        changes = 0
        for i in range(1, img_size-1):
            for j in range(1, img_size - 1):
                neighbors_sum = (image[i - 1][j] + image[i + 1][j] + image[i][j - 1] + image[i][j + 1]) / Temp
                teta1 = np.exp(neighbors_sum - (y[i][j] - 1) ** 2 / 8)
                teta2 = np.exp(-neighbors_sum - (y[i][j] + 1) ** 2 / 8)
                new_pixel = 1 if teta1 > teta2 else -1
                if new_pixel != image[i][j]:
                    changes += 1
                image[i][j] = new_pixel
    return image[1:-1, 1:-1]


# Maximum-likelihood estimate
def gibbs_sampler_ex10_mle(size, y):
    image = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            image[i][j] = 1 if y[i][j] > 0 else -1
    return image

# Sampling all the images
def gibbs_sampler_ex10(Temp):
    size = 100
    sweeps = 50
    image1 = gibbs_sampler_method1(Temp, size, sweeps)
    eta = 2 * np.random.standard_normal(size=(size, size))
    image2 = image1 + eta
    image3 = gibbs_sampler_ex10_3(Temp, size, sweeps, image2)
    image4 = gibbs_sampler_ex10_4(Temp, size, image2)
    image5 = gibbs_sampler_ex10_mle(size, image2)
    return [image1, image2, image3, image4, image5]

# Sampling and displaying all the images for all temp values
def gibbs_sampler_images_display():
    temps = [1, 1.5, 2]
    images = [gibbs_sampler_ex10(temp) for temp in temps]
    f, axarr = plt.subplots(len(temps), 5, figsize=(10, 6))
    cmap = mcolors.ListedColormap(['black', 'white'])
    for i in range(len(temps)):
        axarr[i][0].set_ylabel(f'Temp ={temps[i]}')
        for j in range(5):
            if j != 1:
                img = axarr[i, j].imshow(images[i][j], interpolation="None", vmin=-1, vmax=1, cmap=cmap, aspect='equal')
            else:
                img = axarr[i, j].imshow(images[i][j], interpolation="None", cmap='binary', aspect='equal')
            axarr[i, j].set_xticks([])
            axarr[i, j].set_yticks([])
    cax = f.add_axes([0.92, 0.1, 0.02, 0.8])
    f.colorbar(img, cax=cax, shrink=0.5)
    f.show()





if __name__ == '__main__':
    gibbs_sampler_images_display()
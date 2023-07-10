import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Exercise 1
def G(row_s, Temp):
    return np.exp(np.sum(row_s[:-1] * row_s[1:]) / Temp)


# Exercise 2
def F (row_s, row_t, Temp):
    return np.exp(np.sum(row_s * row_t) / Temp)


# The functions in exercises 3-6 have the following form:
# The first function computes the value of z_temp for a specific value of temp;
# The second function calls the computing function for all the temp values

# Exercise 3
def z_temp(Temp):
    xs = np.array([-1, 1])
    sum = 0
    for x1 in xs:
        for x2 in xs:
            for x3 in xs:
                for x4 in xs:
                    sum += G(np.array([x1,x2]), Temp) * G(np.array([x3,x4]), Temp) * F(np.array([x1,x2]),np.array([x3,x4]),Temp)
    return sum


def compute_ztemp():
    temps = [1, 1.5, 2]
    return [z_temp(temp) for temp in temps]


# Exercise 4
def z_temp_3(Temp):
    xs = np.array([-1, 1])
    sum = 0
    for x1 in xs:
        for x2 in xs:
            for x3 in xs:
                for x4 in xs:
                    for x5 in xs:
                        for x6 in xs:
                            for x7 in xs:
                                for x8 in xs:
                                    for x9 in xs:
                                        sum += G(np.array([x1,x2, x3]), Temp) * G(np.array([x4, x5, x6]), Temp) *\
                                               G(np.array([x7, x8, x9]), Temp)* F(np.array([x1,x2, x3]),np.array([x4, x5, x6]), Temp) *\
                                               F(np.array([x7, x8, x9]), np.array([x4, x5, x6]), Temp)
    return sum


def compute_ztemp_3():
    temps = [1, 1.5, 2]
    return [z_temp_3(temp) for temp in temps]



def y2row(y,width=8):
    """
    y: an integer in (0,...,(2**width)-1)
    """
    if not 0<=y<=(2**width)-1:
         raise ValueError(y)
    my_str=np.binary_repr(y,width=width)
    # my_list = map(int,my_str) # Python 2
    my_list = list(map(int,my_str)) # Python 3
    my_array = np.asarray(my_list)
    my_array[my_array==0]=-1
    row=my_array
    return row

# Exercise 5
def z_temp_markov(Temp):
    ys = np.array([0, 1, 2, 3])
    sum = 0
    for y1 in ys:
        for y2 in ys:
            row_y1 = y2row(y1, 2)
            row_y2 = y2row(y2, 2)
            sum += G(row_y1, Temp) * G(row_y2, Temp) * F(row_y1, row_y2, Temp)
    return sum

def compute_ztemp_markov():
    temps = [1, 1.5, 2]
    return [z_temp_markov(temp) for temp in temps]


# Exercise 6
def z_temp_markov_3(Temp):
    ys = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    sum = 0
    for y1 in ys:
        for y2 in ys:
            for y3 in ys:
                row_y1 = y2row(y1, 3)
                row_y2 = y2row(y2, 3)
                row_y3 = y2row(y3, 3)
                sum += G(row_y1, Temp) * G(row_y2, Temp) * G(row_y3, Temp) * F(row_y1, row_y2, Temp) * F(row_y2, row_y3,Temp)
    return sum



def compute_ztemp_markov_3():
    temps = [1, 1.5, 2]
    return [z_temp_markov_3(temp) for temp in temps]


#  Exercise 7
def T(Temp, size):
    ys = range(0, 2 ** size)
    Ts = np.zeros((size, len(ys)))
    Ts[0] = [1] * len(ys)
    for i in range(1, len(Ts)):
        for y_next in ys:
            for y_curr in ys:
                Ts[i][y_next] += Ts[i-1][y_curr] * G(y2row(y_curr, size), Temp) *\
                                 F(y2row(y_curr, size), y2row(y_next, size), Temp)
    T_last = 0
    for y_curr in ys:
        T_last += Ts[len(Ts) - 1][y_curr] * G(y2row(y_curr, size), Temp)
    return Ts[1:], T_last


def P(Temp, size):
    Ts, last = T(Temp, size)
    ys = range(0, 2 ** size)
    ps = np.ndarray((size-1, len(ys), len(ys)))
    p_last = np.zeros(len(ys))
    for y in ys:
        p_last[y] += Ts[len(Ts) - 1][y] * G(y2row(y, size), Temp) / last
    for k in range(len(ps)-1, 0, -1):
        for y_curr in ys:
            for y_next in ys:
                ps[k][y_next][y_curr] = Ts[k-1][y_curr] * G(y2row(y_curr, size), Temp) *\
                                        F(y2row(y_curr, size), y2row(y_next, size), Temp) / Ts[k][y_next]
    for y_curr in ys:
        for y_next in ys:
            ps[0][y_next][y_curr] = G(y2row(y_curr, size), Temp) * F(y2row(y_curr, size), y2row(y_next, size), Temp) /\
                Ts[0][y_next]
    return ps, p_last




def sample_image(Temp, size, num_of_images):
    ps, p_last = P(Temp, size)
    ys = list(range(0, 2 ** size))
    images = []
    for l in range(num_of_images):
        image = []
        y = np.zeros(size).astype(int)
        y[size - 1] = (np.random.choice(ys, p=p_last))
        for i in range( size-2, -1,  -1):
            y[i] = int(np.random.choice(ys, p=ps[i][y[i+1]]))
        for j in y:
            image.append(y2row(j, size))
        images.append(image)
    return images


def produce_sample():
    temps = [1, 1.5, 2]
    images = [sample_image(temp, 8, 10) for temp in temps]
    f, axarr = plt.subplots(len(temps), 10, figsize=(10, 6))
    cmap = mcolors.ListedColormap(['black', 'white'])
    for i in range(len(temps)):
        axarr[i][0].set_ylabel(f'Temp ={temps[i]}')
        for j in range(10):
            img = axarr[i, j].imshow(images[i][j], interpolation="None", vmin=-1, vmax=1, cmap=cmap, aspect='equal')
            axarr[i, j].set_xticks([])
            axarr[i, j].set_yticks([])
    cax = f.add_axes([0.92, 0.1, 0.02, 0.8])
    f.colorbar(img, cax=cax, shrink=0.5)
    f.show()


# Exercise 8
def produce_sample_10k():
    temps = [1, 1.5, 2]
    img_size = 8
    img_num = 10000
    images = [sample_image(temp, img_size, img_num) for temp in temps]
    Es_12 = [0 for temp in temps]
    Es_18 = [0 for temp in temps]
    for i in range(len(temps)):
        for j in range(img_num):
            Es_12[i] += images[i][j][0][0] * images[i][j][1][1]
            Es_18[i] += images[i][j][0][0] * images[i][j][7][7]
        Es_12[i] /= img_num
        Es_18[i] /= img_num
    return Es_12, Es_18


if __name__ == '__main__':
    produce_sample()
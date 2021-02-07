import numpy as np
import matplotlib.pyplot as plt 

plt.ion()

def lagrange_interpolate(x, y):
    # Matrix where each row is difference vector of x_i - x_j values
    diff = x - x.T

    # Need to remove diagonal for x_i - x_j, i = j
    diff = diff[~np.eye(diff.shape[0],dtype=bool)].reshape(diff.shape[0],-1)

    # Multiply each row for denominator of coefficients
    denom = np.prod(diff, axis=1)

    
    x_i = np.linspace(-1, 1, 100)
    y_i = np.zeros_like(x_i)

    for i in range(len(x_i)):
        # Numerator of Lagrange coefficients
        diff = x_i[i] - x

        # Tile into a matrix
        diff = np.tile(diff.T, (len(x), 1))

        # Remove diagonal for i = j
        diff = diff[~np.eye(diff.shape[0],dtype=bool)].reshape(diff.shape[0],-1)

        # Product of rows are the numerators
        numer = np.prod(diff, axis=1)

        # Compute Lagrange coefficient
        coefficent = numer/denom 

        # Multiply by y values and sum for interpolate
        y_i[i] = (coefficent * y.squeeze()).sum()

    
    return x_i, y_i 

def natural_spline(x, y):
    a = y.copy()[:-1]

    # Assume that x intervals are NOT equispaced (for generality)
    delta_x = x[1:] - x[:-1]
    delta_f = y[1:] - y[:-1]

    R = 3/delta_x[1:] * delta_f[1:] - 3/delta_x[:-1] * delta_f[:-1]

    # Construct the coefficient matrix for c values
    A = np.zeros((R.shape[0], R.shape[0]))
    for i in range(A.shape[0]):
        if i > 0:
            A[i][i-1] = delta_x[i-1]

        A[i][i] = 2 * (delta_x[i] + delta_x[i+1])

        if i + 1 < A.shape[0]: 
            A[i][i+1] = delta_x[i]

    c = np.linalg.inv(A) @ R 
    c = np.vstack([np.array([[0]]), c])

    b = (delta_f[:-1])/delta_x[:-1] - delta_x[:-1]/3 * (2*c[:-1] + c[1:])
    b_last = delta_f[-1]/delta_x[-2] - delta_x[-2]/3 * (2*c[-1] + c[-2])
    b = np.vstack([b, b_last])

    d = 1/(3*delta_x[:-1]) * (c[1:] - c[:-1])
    d_last = c[-1]/(3*delta_x[-1])
    d = np.vstack([d, d_last])


    x_i = np.linspace(-1, 1, 100)
    # y_i = np.zeros_like(x_i)

    # Assuming the intervals are NOT equispaced (for generality)
    interval_index = np.searchsorted(x.squeeze(), x_i, 'right') - 1
    interval_index[-1] = interval_index[-1] - 1
    x_diff = x_i - x[interval_index].squeeze()

    a = a.squeeze()
    b = b.squeeze()
    c = c.squeeze()
    d = d.squeeze()
    y_i = a[interval_index] + b[interval_index]*x_diff + c[interval_index]* x_diff**2 + d[interval_index] * x_diff**3

    # import pdb; pdb.set_trace()
    return x_i, y_i


def plot(x, y, x_i, y_i, title, f):
    plt.figure()
    plt.plot(np.linspace(-1, 1, 100), f(np.linspace(-1, 1, 100)))
    plt.plot(x, y, 'x')
    plt.plot(x_i, y_i)
    plt.title(title)
    plt.legend([r'$f(x)$', r'$f(x_i)$', r'$\hat{f}(x)$'])
    plt.grid()

def plot_error(n, errors):
    plt.figure()
    plt.plot(n, errors, '-o')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Number of intervals (n)')
    plt.ylabel('Max Error')
    plt.title('Natural Cubic Spline Error for $f(x)=\\frac{1}{1+25x^2}, x\in[-1,1]$')
    plt.grid()

def calculate_error(x_i, y_i, f):
    y = f(x_i)

    error = np.abs(y-y_i)
    index = np.argmax(error)
    print('Max Error: {:.4f}, x = {}'.format(error[index], x_i[index]))
    return error[index]


def q1(n):
    f = lambda x: 1/(1+25*x**2)

    x = np.linspace(-1, 1, n)[..., np.newaxis]
    y = f(x)

    x_i, y_i = lagrange_interpolate(x, y)

    title = 'Lagrange Interpolation for $f(x)=\\frac{1}{1+25x^2}, x\in[-1,1]$, n='+str(n)
    plot(x, y, x_i, y_i, title, f)


def q2(n, plotting=True):
    f = lambda x: 1/(1+25*x**2)

    x = np.linspace(-1, 1, n)[..., np.newaxis]
    y = f(x)

    x_i, y_i = natural_spline(x, y)

    title = 'Natural Cubic Spline for $f(x)=\\frac{1}{1+25x^2}, x\in[-1,1]$, n='+str(n)

    if plotting:
        plot(x, y, x_i, y_i, title, f)
    return calculate_error(x_i, y_i, f)

if __name__ == '__main__':
    q1(11)
    q1(21)
    
    error = q2(11)
    error = q2(21)
    
    n_vals = [10, 20, 50, 100, 1000]
    errors = [q2(n+1, plotting=False) for n in n_vals]
    plot_error(n_vals, errors)
    

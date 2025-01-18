# 6D Fractal Explorer

![Python Version](https://img.shields.io/badge/python-3.10-blue)
![PyQt5](https://img.shields.io/badge/PyQt5-%E2%9C%94-brightgreen)
![Numba](https://img.shields.io/badge/Numba-0.60-orange)

Do the infinite details of normal fractals bore you?
Do you want to explore more infinities? Different infinities? Infinite infinities?
Here is a fractal viewer for you.

![viewer](https://github.com/user-attachments/assets/853e2f25-68f5-4a99-81ee-9a0e61f4b2ac)

## Setup

`pip install -r requirements.txt`

`python main.py`

## 6D Fractals

This explorer treats the 3 complex numbers ![z_0, c, p ∈ ℂ](https://latex.codecogs.com/png.latex?z_0%2C%20c%2C%20p%20%5Cin%20%5Cmathbb%7BC%7D) used in the fractal equation ![z_(n+1) = z_n ^ p + c](https://latex.codecogs.com/png.latex?z_{n+1}=z_n^p+c) as a 6D space ![\mathbf{v} \in \mathbb{R}^6,\ \mathbf{v} = [c_a, c_b, z_{0a}, z_{0b}, p_a,
p_b]^T](https://latex.codecogs.com/png.latex?%5Cmathbf%7Bv%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E6%2C%20%5C%5C%20%5Cmathbf%7Bv%7D%20%3D%20%5B%20c_a%2C%20c_b%2C%20z_%7B0a%7D%2C%20z_%7B0b%7D%2C%20p_a%2C%20p_b%20%5D%5ET).
It lets you explore any 2D slice through this 6D parameter space. 
Such a slice is rastered into a matrix (w, h, 6) with (w, h) being the image dimensions.
For each future pixel, there exists a set of 6 parameters.
Those parameters are plugged in the equation and run until ![z](https://latex.codecogs.com/png.latex?z) grows larger than some threshold, or a max number of iterations is reached.
The pixel is colored according to how many iterations it took.

Any random plane spanned by 3 points ![\mathbf{v}, \mathbf{o}, \mathbf{u} \in \mathbb{R}^6](https://latex.codecogs.com/png.latex?%5Cmathbf%7Bv%7D%2C%20%5Cmathbf%7Bo%7D%2C%20%5Cmathbf%7Bu%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E6) (Top, Center, Left), gives you a fractal.:

![\mathbf{o} = \begin{bmatrix} 0.8066 \\ 0.6864 \\ 0.0024 \\ 0.7232 \\ 1.5890 \\ 1.4074 \end{bmatrix}, \quad
\mathbf{u} = \begin{bmatrix} 0.5130 \\ -0.7672 \\ -0.3266 \\ -0.6291 \\ -1.7435 \\ 0.1312 \end{bmatrix}, \quad
\mathbf{v} = \begin{bmatrix} 0.5229 \\ 1.0324 \\ -0.1480 \\ -0.1900 \\ 0.3907 \\ 1.5534 \end{bmatrix}](https://latex.codecogs.com/png.latex?%5Cmathbf%7Bo%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%200.8066%20%5C%5C%200.6864%20%5C%5C%200.0024%20%5C%5C%200.7232%20%5C%5C%201.5890%20%5C%5C%201.4074%20%5Cend%7Bbmatrix%7D%2C%20%5Cquad%20%5Cmathbf%7Bu%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%200.5130%20%5C%5C%20-0.7672%20%5C%5C%20-0.3266%20%5C%5C%20-0.6291%20%5C%5C%20-1.7435%20%5C%5C%200.1312%20%5Cend%7Bbmatrix%7D%2C%20%5Cquad%20%5Cmathbf%7Bv%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%200.5229%20%5C%5C%201.0324%20%5C%5C%20-0.1480%20%5C%5C%20-0.1900%20%5C%5C%200.3907%20%5C%5C%201.5534%20%5Cend%7Bbmatrix%7D)

![random](https://github.com/user-attachments/assets/14810d40-0a21-41c1-9cb1-d0bc2dc67e4e)

Each with its own infinite detail surface. Below is a zoomed-in view of the above fractal:

![random_zoomed](https://github.com/user-attachments/assets/d9f96c9f-9f6d-423a-9400-caee2f19891d)

Of course there are a few "classic" slices ( ͡° ͜ʖ ͡°). Here's the [Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set):

![\mathbf{o} = \begin{bmatrix} 0.0 \\ 0.0 \\ 0.0 \\ 0.0 \\ 2.0 \\ 0.0 \end{bmatrix}, \quad
\mathbf{u} = \begin{bmatrix} 1.0 \\ 0.0 \\ 0.0 \\ 0.0 \\ 2.0 \\ 0.0 \end{bmatrix}, \quad
\mathbf{v} = \begin{bmatrix} 0.0 \\ 1.0 \\ 0.0 \\ 0.0 \\ 2.0 \\ 0.0 \end{bmatrix}](https://latex.codecogs.com/png.latex?%5Cmathbf%7Bo%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%200.0%20%5C%5C%200.0%20%5C%5C%200.0%20%5C%5C%200.0%20%5C%5C%202.0%20%5C%5C%200.0%20%5Cend%7Bbmatrix%7D%2C%20%5Cquad%20%5Cmathbf%7Bu%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%201.0%20%5C%5C%200.0%20%5C%5C%200.0%20%5C%5C%200.0%20%5C%5C%202.0%20%5C%5C%200.0%20%5Cend%7Bbmatrix%7D%2C%20%5Cquad%20%5Cmathbf%7Bv%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%200.0%20%5C%5C%201.0%20%5C%5C%200.0%20%5C%5C%200.0%20%5C%5C%202.0%20%5C%5C%200.0%20%5Cend%7Bbmatrix%7D)

![Mandelbrot Set](https://github.com/user-attachments/assets/ea6d3fa6-6db2-4ba0-b90c-82392abe5a7d)

And a (Julia set)[https://en.wikipedia.org/wiki/Julia_set] with ![c = 0.45 + 0.14i](https://latex.codecogs.com/png.latex?c%20%3D%200.45%20%2B%200.14i):

![\mathbf{o} = \begin{bmatrix} 0.45 \\ 0.14 \\ 0.00 \\ 0.00 \\ 2.00 \\ 0.00 \end{bmatrix}, \quad
\mathbf{u} = \begin{bmatrix} 0.45 \\ 0.14 \\ 1.00 \\ 0.00 \\ 2.00 \\ 0.00 \end{bmatrix}, \quad
\mathbf{v} = \begin{bmatrix} 0.45 \\ 0.14 \\ 0.00 \\ 1.00 \\ 2.00 \\ 0.00 \end{bmatrix}](https://latex.codecogs.com/png.latex?%5Cmathbf%7Bo%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%200.45%20%5C%5C%200.14%20%5C%5C%200.00%20%5C%5C%200.00%20%5C%5C%202.00%20%5C%5C%200.00%20%5Cend%7Bbmatrix%7D%2C%20%5Cquad%20%5Cmathbf%7Bu%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%200.45%20%5C%5C%200.14%20%5C%5C%201.00%20%5C%5C%200.00%20%5C%5C%202.00%20%5C%5C%200.00%20%5Cend%7Bbmatrix%7D%2C%20%5Cquad%20%5Cmathbf%7Bv%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%200.45%20%5C%5C%200.14%20%5C%5C%200.00%20%5C%5C%201.00%20%5C%5C%202.00%20%5C%5C%200.00%20%5Cend%7Bbmatrix%7D)

![Julia Set](https://github.com/user-attachments/assets/de744e45-d6f6-46f6-a576-db9fec93aa57)

You can even mix them. Here's a bit of the Julia set above and the Mandelbrot set:

![\mathbf{o} = \begin{bmatrix} 0.45 \\ 0.14 \\ 0.0 \\ 0.0 \\ 2.0 \\ 0.0 \end{bmatrix}, \quad
\mathbf{u} = \begin{bmatrix} 1.0 \\ 0.0 \\ 1.0 \\ 0.0 \\ 2.0 \\ 0.0 \end{bmatrix}, \quad
\mathbf{v} = \begin{bmatrix} 0.0 \\ 1.0 \\ 0.0 \\ 1.0 \\ 2.0 \\ 0.0 \end{bmatrix}](https://latex.codecogs.com/png.latex?%5Cmathbf%7Bo%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%200.45%20%5C%5C%200.14%20%5C%5C%200.0%20%5C%5C%200.0%20%5C%5C%202.0%20%5C%5C%200.0%20%5Cend%7Bbmatrix%7D%2C%20%5Cquad%20%5Cmathbf%7Bu%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%201.0%20%5C%5C%200.0%20%5C%5C%201.0%20%5C%5C%200.0%20%5C%5C%202.0%20%5C%5C%200.0%20%5Cend%7Bbmatrix%7D%2C%20%5Cquad%20%5Cmathbf%7Bv%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%200.0%20%5C%5C%201.0%20%5C%5C%200.0%20%5C%5C%201.0%20%5C%5C%202.0%20%5C%5C%200.0%20%5Cend%7Bbmatrix%7D)

![mix](https://github.com/user-attachments/assets/acb04112-28ef-4ec7-8beb-b8956268f553)

## Interface


### Viewport

- Rectangle-select regions you want to zoom into.

### Settings

- Save and load the settings of the current fractal as files in YAML format.
- Export images at a given resolution.
- Go back in history or reset to the initial view.
- Change the color of fractal and UI.

### Movement

- Move, rotate, and scale your current view in the current fractal plane.
- Modify those values directly. Confirm with Enter.

### Translation

- Translate the whole fractal plane in 6D along any axis.

### Rotation

- Rotate the entire fractal plane around any two axes.

### Parameters

- Modify parameters directly. Confirm with Enter.
- Randomize all parameters.
- Perturb the current parameter settings.
- To exempt any combination of point and/or dimension from being randomized or perturbed, toggle their respective headers in the parameter view.

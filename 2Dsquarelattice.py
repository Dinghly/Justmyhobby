import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define the symbols
kx, ky, a, r1, r2 = sp.symbols('kx ky a r1 r2')

# Define the matrix A
A = sp.Matrix([
    [4*r1*sp.sin(kx*a/2)**2 + 4*r2*(1 - sp.cos(kx*a)*sp.cos(ky*a)), 4*r2*sp.sin(kx*a)*sp.sin(ky*a)],
    [4*r2*sp.sin(kx*a)*sp.sin(ky*a), 4*r1*sp.sin(ky*a/2)**2 + 4*r2*(1 - sp.cos(kx*a)*sp.cos(ky*a))]
])

# Calculate the eigenvalues
eigenvalues = A.eigenvals()
# print(eigenvalues)
# Extract w^2 from the eigenvalues
w2_1 = list(eigenvalues.keys())[0]
w2_2 = list(eigenvalues.keys())[1]

# Define the function w = sqrt(w2)
w_1 = sp.sqrt(w2_1)
w_2 = sp.sqrt(w2_2)
print(w_2)

# Function to evaluate w given kx and ky
w_function_1 = sp.lambdify((kx, ky, a, r1, r2), w_1, 'numpy')
w_function_2 = sp.lambdify((kx, ky, a, r1, r2), w_2, 'numpy')

############################################
# Define constants                         #
a_val = 1  # lattice constant              #
r1_val = 3  # example value for r1         #
r2_val = 0.5  # example value for r2         #
############################################

# Define kx and ky ranges
kx_vals = np.linspace(0, np.pi, 100)
ky_vals = np.linspace(0, np.pi, 100)

# Calculate w(kx, ky=0)
w_1_kx_ky0 = [w_function_1(kx_val, 0, a_val, r1_val, r2_val) for kx_val in kx_vals]
w_2_kx_ky0 = [w_function_2(kx_val, 0, a_val, r1_val, r2_val) for kx_val in kx_vals]

# Calculate w(kx=pi, ky)
w_1_kxpi_ky = [w_function_1(np.pi, ky_val, a_val, r1_val, r2_val) for ky_val in ky_vals]
w_2_kxpi_ky = [w_function_2(np.pi, ky_val, a_val, r1_val, r2_val) for ky_val in ky_vals]

# Calculate w(kx=ky)
w_1_kxky = [w_function_1(k_val, k_val, a_val, r1_val, r2_val) for k_val in kx_vals]
w_2_kxky = [w_function_2(k_val, k_val, a_val, r1_val, r2_val) for k_val in kx_vals]

plt.figure(figsize=(12, 6))

# Plot w(kx) for ky=0
plt.subplot(1, 3, 1)
plt.plot(kx_vals, w_1_kx_ky0)
plt.plot(kx_vals, w_2_kx_ky0)
#plt.title('w(kx) for ky = 0')
plt.xlabel('$\Gamma$ to X')
plt.ylabel('w')
plt.ylim(0, 15)
plt.grid(True)

# Plot w(ky) for kx=pi
plt.subplot(1, 3, 2)
plt.plot(ky_vals, w_1_kxpi_ky)
plt.plot(ky_vals, w_2_kxpi_ky)
plt.title('Phonon dispersion relation of square lattice')
plt.xlabel('X to M')
#plt.ylabel('w')
plt.ylim(0, 15)
plt.grid(True)

# Plot w(kx=ky) and flip x-axis
plt.subplot(1, 3, 3)
plt.plot(kx_vals[::-1], w_1_kxky)
plt.plot(kx_vals[::-1], w_2_kxky)
#plt.title('w(kx = ky)')
plt.xlabel('M to $\Gamma$')
#plt.ylabel('w')
plt.ylim(0, 15)
plt.grid(True)
plt.legend(title=f'$r_1 = {r1_val}$, $r_2 = {r2_val}$')

# Display the plot
plt.subplots_adjust(wspace=0)
plt.show()
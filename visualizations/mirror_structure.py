import matplotlib.pyplot as plt

ILE_LUSTER = [
    9,      # row 0
    11,     # row 1
    13,     # row 2
    15,     # row 3
    17,     # row 4
    17,     # row 5
    17,     # row 6
    17,     # row 7
    17,     # row 8
    17,     # row 9
    17,     # row 10
    17,     # row 11
    17,     # row 12
    15,     # row 13
    13,     # row 14
    11,     # row 15
    9       # row 16
]

# Y - współrzędne pionowe (od góry do dołu)
Y = list(range(16, -1, -1))  # [16, 15, 14, ..., 1, 0]

# X - przesunięcie dla wycentrowania każdego rzędu
# max szerokość = 17, więc offset = (17 - n) / 2
X = [(17 - n) // 2 for n in ILE_LUSTER]

for n, x, y in zip(ILE_LUSTER, X, Y):
    for i in range(n):
        plt.scatter(i + x, y)

plt.axis('equal')
plt.title(f'Struktura teleskopu ({sum(ILE_LUSTER)} luster)')
plt.show()
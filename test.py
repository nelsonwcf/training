#!/bin/python3

import math
import os
import random
import re
import sys


# Complete the cavityMap function below.
def cavityMap(grid):
    n = len(grid)
    if n <= 2:
        return grid
    for lin in range(n):
        for col in range(n):
            markCavity(grid, lin, col)


def markCavity(g, lin, col):
    if lin - 1 < 0 or lin + 1 == len(g) or col - 1 < 0 or col + 1 == len(g):
        return

    v = int(g[lin][col])
    if g[lin - 1][col] == 'X' or g[lin + 1][col] == 'X' or g[lin][col - 1] == 'X' or g[lin][col + 1] == 'X':
        return

    if v > int(g[lin - 1][col]) and v > int(g[lin + 1][col]) and v > int(g[lin][col - 1]) and v > int(
            g[lin][col + 1]):
        g[lin][col] = 'X'
    return


if __name__ == '__main__':
    n = int(input())

    grid = []

    for _ in range(n):
        grid_item = input()
        grid.append(grid_item)

    print(grid)
    result = cavityMap(grid)

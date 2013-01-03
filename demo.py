#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as pl

import triangle


if __name__ == "__main__":
	data = np.random.randn(5000 * 4).reshape([4, 5000])
	extents = [[-i, i] for i in range(1, 5)]
	triangle.corner(data, labels=["var 1", "var 2", "var 3", "var 4"],
					extents=extents)
	pl.savefig("demo.png")

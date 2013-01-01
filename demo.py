#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as pl

import triangle


if __name__ == "__main__":
	data = np.random.randn(5000 * 4).reshape([4, 5000])
	extents = [[-i, i] for i in range(1, 5)]
#	triangle.corner(data, labels=["var 1", "var 2", "var 3", "var 4"],
#					extents=extents)
#	pl.savefig("demo.png")

	fig = pl.figure()
	triangle.hist2d(data[0], data[1], ax=fig.gca(), extent=[extents[3], extents[3]])
	fig.savefig("/Users/geoff/Desktop/demo1.png")
	
	fig = pl.figure()
	triangle.hist2d(data[0], data[1], ax=fig.gca(), extent=[extents[2], extents[3]])
	fig.savefig("/Users/geoff/Desktop/demo2.png")
	
	fig = pl.figure()
	triangle.hist2d(data[0], data[1], ax=fig.gca(), extent=[extents[1], extents[3]])
	fig.savefig("/Users/geoff/Desktop/demo3.png")

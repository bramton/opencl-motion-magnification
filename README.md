# OpenCL accelerated motion magnification
This code provides an OpenCL accelerated phase-based motion magnification implementation. The work implements the paper "Riesz Pyramids for Fast Phase-Based Video Magnification".

# Dependencies
* OpenCV (I used 4.5.1)
* [Argtable](https://www.argtable.org) for parsing command-line arguments

# Tips & tricks
* Use uncompressed video if available
* Increase sigma for larger videos

# Limitations/improvements
* Only single octave pyramid supported, that is area between adjacent levels of the pyramid decrease by a factor four
* Use [Halide](https://halide-lang.org) for the implementation
* Currently a Butterworth bandpass filter is used. Perhaps a different type, e.g. Chebyshev type II could improve the temporal phase filtering

# Acknowledgements
The pseudo code and the actual code provided by the authors on their [website](http://people.csail.mit.edu/nwadhwa/riesz-pyramid/) have been indispensable for creating this implementation.

# Legal stuff
The original code is patented (US20150195430A1) but is free to use for research and non-commercial purposes. Ye be warned.
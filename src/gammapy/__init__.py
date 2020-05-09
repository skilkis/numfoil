"""Defines the matplotlib style for all files."""

import matplotlib
from matplotlib import pyplot as plt

plt.style.use("ggplot")  # Sets style-sheet globally

STYLE_CONFIG = {}

matplotlib.rcParams.update(STYLE_CONFIG)

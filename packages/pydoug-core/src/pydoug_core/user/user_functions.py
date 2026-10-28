"""
Add your own functions to PyDoug using the template herein.
"""

# Imports

# from typing import Annotated

# import numpy as np

# from pydoug_core.registry import register_operation


# Functions

# @register_operation(
#     category="user",
#     display_name="Simple Threshold",
# )
# def sample_function(
#     imarray: np.ndarray,
#     Thresholds: tuple[
#         Annotated[int, {"max": 65535, "min": 0}],
#         Annotated[int, {"max": 65535, "min": 0}]
#     ] = (50, 200),
# ) -> np.ndarray:
#     """
#     This is a nympy-style docstring for a sample function that thresholds an image
#     using inclusive upper and lower bounds and returns a binary image array where all
#     pixels within the threshold range are white and all pixels outside the range are
#     black.
#
#     Parameters
#     ----------
#     imarray: np.ndarray
#         The image array to be thresholded.
#     Thresholds: tuple[int, int]
#         The intensity values to use for the threshold range.
#
#     Returns
#     -------
#     thresh_array: np.ndarray
#         The thresholded image array.
#     """
#
#     thresh_array: np.ndarray = np.zeros(imarray.shape, np.uint8)
#     thresh_array[(imarray <= max(Thresholds)) & (imarray >= min(Thresholds))] = 255
#     return thresh_array

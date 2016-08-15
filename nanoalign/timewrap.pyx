#cython: boundscheck=False
#cython: wraparound=False

"""
Time series analysis
"""

import numpy as np
from cpython cimport array
from libc.stdlib cimport malloc, free
import array


def edr_distance(seq_1, seq_2):
    #return cedr_distance(seq_1, seq_2)
    return simple_distance(seq_1, seq_2)


cdef float simple_distance(seq_1, seq_2):
    cdef float score = 0
    cdef float THRESHOLD = 1
    cdef float diff = 0
    for i in range(len(seq_1)):
        diff = abs(seq_1[i] - seq_2[i])
        if diff > THRESHOLD:
            score += diff * diff
    return score


cdef float cedr_distance(seq_1, seq_2):
    """
    EDR: Edit distance on real sequences
    """
    cdef float GAP = 500
    cdef float THRESHOLD = 1
    #cdef int MISS = 1
    #cdef int MATCH = 0

    cdef int len_1 = len(seq_1) + 1
    cdef int len_2 = len(seq_2) + 1
    cdef int i = 0
    cdef int j = 0

    cdef float match_score = 0
    cdef float cross = 0
    cdef float left = 0
    cdef float up = 0
    cdef float max_score = 0

    cdef float *score_matrix = <float*>malloc(len_1 * len_2 * sizeof(float))
    cdef float *arr_1 = <float*>malloc((len_1 - 1) * sizeof(float))
    for i in range(0, len_1 - 1):
        arr_1[i] = seq_1[i]
    cdef float *arr_2 = <float*>malloc((len_2 - 1)* sizeof(float))
    for i in range(0, len_2 - 1):
        arr_2[i] = seq_2[i]

    score_matrix[0] = 0
    for i in range(1, len_1):
        score_matrix[i + 0 * len_2] = score_matrix[(i - 1) + 0 * len_2] + GAP
    for i in range(1, len_2):
        score_matrix[0 + i * len_2] = score_matrix[0 + (i - 1) * len_2] + GAP

    for i in range(1, len_1):
        for j in range(1, len_2):
            if abs(arr_1[i - 1] - arr_2[j - 1]) < THRESHOLD:
                match_score = 0
            else:
                match_score = (arr_1[i - 1] - arr_2[j - 1]) * (arr_1[i - 1] - arr_2[j - 1])
            cross = score_matrix[(i - 1) + (j - 1) * len_2] + match_score

            left = score_matrix[i + (j - 1) * len_2] + GAP
            up = score_matrix[(i - 1) + j * len_2] + GAP

            max_score = min(min(left, up), cross)
            score_matrix[i + j * len_2] = max_score

    #cdef float score = score_matrix[len_1 * len_2 - 1]

    free(score_matrix)
    free(arr_1)
    free(arr_2)

    return max_score

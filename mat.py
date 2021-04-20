import numpy as np


class mat(np.ndarray):
    """ new matrix class based on np.ndarray.
        overwrites the multiplication operator so that
        it performs matrix multiplication by default
        instead of the standard scalar multiplcation.
        This was done because np.matrix is being
        deprecated """
    def __new__(cls, input_array, info=None):
        """ inherits from standard np.ndarray """
        obj = np.asarray(input_array).view(cls)
        obj.info = info
        return obj

    def __init__(self, *args, **kwargs):
        """ call base __init__ and fix shape attribute for
        row vectors """
        super(mat, self).__init__(*args, **kwargs)
        if len(self.shape) == 1:
            dim = self.shape[0]
            self.shape = (1, dim)

    def __mul__(self, other):
        """ set matrix multiplication as default """
        if isinstance(other, float) or isinstance(other, int):
            return super(mat, self).__mul__(other)
        else:
            return np.matmul(self, other)

    def __getitem__(self, other):
        """ fix vector dimensions when slicing """
        if isinstance(other, tuple):
            try:
                if isinstance(other[0], slice):
                    ret = super(mat, self).__getitem__(other)
                    if len(ret.shape) == 1:
                        dim = ret.shape[0]
                        ret.shape = (dim, 1)
                    return ret
                elif isinstance(other[1], slice):
                    ret = super(mat, self).__getitem__(other)
                    if len(ret.shape) == 1:
                        dim = ret.shape[0]
                        ret.shape = (1, dim)
                    return ret
                else:
                    return super(mat, self).__getitem__(other)
            except IndexError:
                return super(mat, self).__getitem__(other)
        else:
            return super(mat, self).__getitem__(other)

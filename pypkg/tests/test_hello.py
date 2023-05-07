import numpy

from hello import return_two, MH

def test_return_two():
    assert return_two() == 2


def test_MH():
    numpy.testing.assert_array_equal(numpy.eye(10), MH(10).A)
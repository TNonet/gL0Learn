from typing import Union

from gl0learn.bounds import Bounds
from gl0learn.penalty import Penalty
from gl0learn.utils import overlaps, ClosedInterval, intersect


class Oracle:
    def __init__(self, penalty: Penalty, bounds: Bounds):
        self.penalty = penalty
        self.bounds = bounds

        if not overlaps(self.bounds.num_features, self.penalty.num_features):
            raise ValueError("expected Bounds and Penalty to have overlapping number of features, but are not.")

    @property
    def num_features(self) -> Union[ClosedInterval, int]:
        return intersect(self.penalty.num_features, self.bounds.num_features)

    def __repr__(self):
        return f"Oracle(penalty={self.penalty}, bounds={self.bounds})"

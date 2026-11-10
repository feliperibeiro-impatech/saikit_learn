from typing import Optional

class TreeNode:
    def __init__(
        self,
        feature: Optional[int] = None,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None,
        value: Optional[float] = None,
        gain: Optional[float] = None,
        threshold: Optional[float] = None,
        n_samples: int = 0,
    ):
        self.feature = feature
        self.left = left
        self.right = right
        self.value = value
        self.gain = gain
        self.threshold = threshold
        self.n_samples = n_samples

    @property
    def is_leaf(self) -> None:
        return self.feature is None
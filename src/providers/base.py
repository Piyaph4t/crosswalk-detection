from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np

class VisionProvider(ABC):
    """
    Abstract Base Class for all vision input providers.
    Ensures a consistent interface for frame retrieval.
    """

    @abstractmethod
    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Retrieve the next frame from the source.

        Returns:
            Tuple of (frame, frame_id).
            Returns (None, None) when the source is exhausted.
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Release hardware or file resources."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

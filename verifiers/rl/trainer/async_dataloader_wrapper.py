import threading
from collections import deque
from typing import Any

from torch.utils.data import DataLoader  # type: ignore[unresolved-import]


class AsyncDataLoaderWrapper:
    """
    Wraps a DataLoader to provide batch prefetching capabilities for async generation.

    This wrapper maintains a buffer of upcoming batches that can be accessed
    without advancing the main iterator, allowing async generation to work
    ahead while training continues on current batches.
    """

    def __init__(self, dataloader: DataLoader, buffer_size: int = 5):
        self.dataloader = dataloader
        self.buffer_size = buffer_size
        self._buffer = deque(maxlen=buffer_size)
        self._current_iterator = None  # Iterator for current epoch
        self._next_iterator = None  # Iterator for next epoch (created when needed)
        self._lock = threading.Lock()
        self._exhausted = False
        self._current_batch = None  # Track the current batch

    def __iter__(self):
        """Reset and return iterator"""
        with self._lock:
            # If we pre-created an iterator for the next epoch, use it
            if self._next_iterator is not None:
                self._current_iterator = self._next_iterator
                self._next_iterator = None
            else:
                self._current_iterator = iter(self.dataloader)

            self._buffer.clear()
            self._exhausted = False
            self._current_batch = None
        return self

    def __next__(self):
        """Get next batch, refilling buffer as needed"""
        with self._lock:
            # If buffer is empty, try to fill it
            if not self._buffer and not self._exhausted:
                self._fill_buffer()

            if not self._buffer:
                raise StopIteration

            # Store current batch before returning
            self._current_batch = self._buffer.popleft()
            return self._current_batch

    def peek_ahead(self, n: int = 1) -> list[Any]:
        """
        Peek at the next n batches without consuming them.
        If n=0, returns the current batch (if available).
        Returns fewer batches if not enough are available.
        """
        with self._lock:
            if n == 0:
                # Return current batch if available
                return [self._current_batch] if self._current_batch is not None else []

            # Ensure buffer has enough items
            while len(self._buffer) < n and not self._exhausted:
                self._fill_buffer_single()

            # Return up to n items from buffer
            return list(self._buffer)[:n]

    def _fill_buffer(self):
        """Fill the buffer up to buffer_size"""
        while len(self._buffer) < self.buffer_size and not self._exhausted:
            self._fill_buffer_single()

    def _fill_buffer_single(self):
        """Add a single batch to the buffer"""
        # Initialize current iterator if needed
        if self._current_iterator is None:
            self._current_iterator = iter(self.dataloader)

        try:
            # Try to get batch from current iterator
            batch = next(self._current_iterator)
            self._buffer.append(batch)
        except StopIteration:
            # Current epoch exhausted - try to create iterator for next epoch
            if self._next_iterator is None:
                try:
                    self._next_iterator = iter(self.dataloader)
                except Exception:
                    # Can't create new iterator, we're done
                    self._exhausted = True
                    return

            # Try to get batch from next epoch's iterator
            try:
                batch = next(self._next_iterator)
                self._buffer.append(batch)
            except StopIteration:
                # Next iterator also exhausted, we're truly done
                self._exhausted = True

    def get_future_batches(self, start_offset: int, count: int) -> list[Any]:
        """
        Get future batches starting from start_offset positions ahead.
        This is used by async generation to get batches for future steps.

        Args:
            start_offset: How many batches ahead to start
            count: Number of batches to return

        Returns:
            list of batches (may be fewer than requested if not available)
        """
        with self._lock:
            # Ensure we have enough batches in buffer
            needed = start_offset + count
            while len(self._buffer) < needed and not self._exhausted:
                self._fill_buffer_single()

            # Extract the requested range
            result = []
            for i in range(start_offset, min(start_offset + count, len(self._buffer))):
                result.append(self._buffer[i])

            return result

    def __len__(self):
        """Return length of underlying dataloader if available"""
        return len(self.dataloader)

    @property
    def batch_size(self):
        """Return batch size of underlying dataloader"""
        return self.dataloader.batch_size

    @property
    def dataset(self):
        """Return dataset of underlying dataloader"""
        return self.dataloader.dataset

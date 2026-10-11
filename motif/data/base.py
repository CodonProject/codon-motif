from typing import Any, Dict, Iterator, Protocol, runtime_checkable


@runtime_checkable
class Stateful(Protocol):
    '''
    Structural protocol for objects that persist an internal cursor.

    Anything implementing both `state_dict()` and `load_state_dict()` satisfies
    it: `isinstance(obj, Stateful)` is decided by method name at run time, so no
    explicit inheritance is required.

    Implementing only half of it - `state_dict()` alone, for example - does not
    count: the pipeline then raises immediately instead of silently feeding the
    data from the beginning again after a resume.
    '''

    def state_dict(self) -> Dict[str, Any]:
        '''
        Returns:
            Dict[str, Any]: A picklable state dictionary.
        '''
        ...

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        '''
        Args:
            state (Dict[str, Any]): State previously produced by `state_dict()`.
        '''
        ...


def is_stateful(obj: Any) -> bool:
    '''
    Whether `obj` fully implements Stateful (both methods present).

    Args:
        obj (Any): Object to test.

    Returns:
        bool: True when both protocol methods exist.
    '''
    return isinstance(obj, Stateful)


class MotifBasicDataset:
    '''Common root of every dataset: used for type grouping only, no behaviour.'''


class MotifDataset(MotifBasicDataset):
    '''
    Map-style dataset: has a length and supports random access by index
    (`TextFileDataset` / `MotifSFT`).

    Random access is what makes a resume cheap: the upstream can seek directly, so
    `ChunkedTokenStream` restarts from `upstream_offset` instead of decoding the
    preceding records again.
    '''

    def __len__(self) -> int:
        '''
        Returns:
            int: Number of records.

        Raises:
            NotImplementedError: Always; subclasses must implement this.
        '''
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Any:
        '''
        Args:
            idx (int): Record index.

        Returns:
            Any: The record at that index.

        Raises:
            NotImplementedError: Always; subclasses must implement this.
        '''
        raise NotImplementedError


class MotifIterableDataset(MotifBasicDataset):
    '''
    Iterable-style dataset: can only be streamed in order, seeking through
    `iter_from(offset)`.

    Subclasses must implement `iter_from` and encode "how far we have consumed"
    into their own state, so that after `state_dict()` is restored
    `iter_from(0)` still continues from the correct position.
    '''

    def iter_from(self, offset: int) -> Iterator[Any]:
        '''
        Args:
            offset (int): Lower bound of the upstream record offset (max'ed with
                the dataset's own cursor).

        Returns:
            Iterator[Any]: The data stream starting at that position.
        '''
        raise NotImplementedError

    def __iter__(self) -> Iterator[Any]:
        '''
        Returns:
            Iterator[Any]: The stream starting at offset 0 (the subclass cursor still
            applies).
        '''
        return self.iter_from(0)

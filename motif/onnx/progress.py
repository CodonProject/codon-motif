import sys
import threading
import time

SPINNER = ('|', '/', '-', '\\')


def format_seconds(seconds):
    '''`3.4s`, `1m04.1s`, `1h02m03s`.'''
    if seconds < 60.0:
        return f'{seconds:.1f}s'
    minutes, rest = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f'{int(minutes)}m{rest:04.1f}s'
    hours, minutes = divmod(int(minutes), 60)
    return f'{hours}h{minutes:02d}m{rest:02.0f}s'


class Phase:
    '''One phase of a `Progress`: a context manager that can report its own units.'''

    def __init__(self, bar, name, weight, units=None):
        self.bar = bar
        self.name = name
        self.weight = float(weight)
        self.units = None if units is None else int(units)
        self.done = 0
        self.detail = ''
        self.started = 0.0

    def __enter__(self):
        self.started = time.perf_counter()
        self.bar._enter_phase(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.bar._leave_phase(self)
        return False

    @property
    def elapsed(self):
        '''Seconds since the phase was entered.'''
        return time.perf_counter() - self.started

    def advance(self, count=1, detail=None):
        '''Mark `count` more units of this phase as done, e.g. `advance(1, '3/8 steps')`.'''
        self.done += count
        if detail is not None:
            self.detail = str(detail)
        self.bar._render()
        return self

    def note(self, text):
        '''Print a line above the bar without disturbing it.'''
        self.bar.note(text)


class Progress:
    '''
    A weighted bar over a fixed list of phases.

    Args:
        phases (Sequence[Tuple[str, float]]): `(name, weight)` per phase, in order.
        label (str): Prefix of every line.
        stream (Optional[TextIO]): Where to draw; `sys.stderr` by default.
        enabled (bool): When False no bar is drawn (notes still print).
        width (int): Bar width in characters.
        tick (float): Redraw interval while a phase runs.

    On a terminal the bar is redrawn in place and every finished phase leaves a
    line behind; when the stream is not a terminal the phases print one line
    each, which keeps logs and CI output readable.
    '''

    def __init__(self, phases, label='progress', stream=None, enabled=True,
                 width=28, tick=0.5):
        self.phases = [(str(name), float(weight)) for name, weight in phases]
        self.label = str(label)
        self.stream = sys.stderr if stream is None else stream
        self.width = int(width)
        self.tick = float(tick)
        self.enabled = bool(enabled)
        self.total = sum(weight for _, weight in self.phases) or 1.0
        self.tty = bool(getattr(self.stream, 'isatty', lambda: False)())
        self.current = None
        self.done = 0.0
        self.started = 0.0
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._spin = 0
        self._drawn = 0

    def __enter__(self):
        self.started = time.perf_counter()
        if self.enabled and self.tty and self.tick > 0:
            self._thread = threading.Thread(target=self._loop, name='progress-ticker',
                                            daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.current = None
        if self.enabled and self.tty:
            if exc_type is None:
                self.done = self.total
                self._draw()
                self._write('\n')
            else:
                self._write('\n')
        elif self.enabled and exc_type is None:
            self._write(f'[{self.label}] stopped after '
                        f'{format_seconds(time.perf_counter() - self.started)}\n')
        return False

    def phase(self, name, units=None):
        '''A `Phase` context manager for `name`, counting `units` when given.'''
        return Phase(self, name, self._weight_of(name), units)

    def note(self, text):
        '''Print one line; on a terminal it lands above the bar.'''
        text = str(text)
        with self._lock:
            if self.tty:
                if self.enabled:
                    self._write('\r' + ' ' * self._drawn + '\r')
                    self._drawn = 0
                self._write(text + '\n')
                self._render()
            else:
                self._write(text + '\n')

    def _weight_of(self, name):
        for candidate, weight in self.phases:
            if candidate == name:
                return weight
        return 1.0

    def _index_of(self, name):
        for index, (candidate, _) in enumerate(self.phases, start=1):
            if candidate == name:
                return index
        return len(self.phases)

    def _enter_phase(self, phase):
        with self._lock:
            self.current = phase
            self._render()

    def _leave_phase(self, phase):
        with self._lock:
            self.done = min(self.total, self.done + phase.weight)
            self.current = None
            line = (f'[{self.label}] {self._index_of(phase.name)}/{len(self.phases)} '
                    f'{phase.name}  {format_seconds(phase.elapsed)}  '
                    f'({100.0 * self.done / self.total:3.0f}%)')
            if self.enabled and self.tty:
                self._write('\r' + ' ' * self._drawn + '\r')
                self._drawn = 0
                self._write(line + '\n')
            elif self.enabled:
                self._write(line + '\n')
            self._render()

    def _fraction(self):
        if self.current is None:
            return min(1.0, self.done / self.total)
        portion = 1.0
        if self.current.units:
            portion = min(1.0, self.current.done / float(self.current.units))
        return min(1.0, (self.done + self.current.weight * portion) / self.total)

    def _line(self):
        fraction = self._fraction()
        filled = int(round(fraction * self.width))
        bar = '#' * filled + '.' * (self.width - filled)
        elapsed = format_seconds(time.perf_counter() - self.started)
        if self.current is None:
            return (f'{self.label} [{bar}] {100.0 * fraction:3.0f}% '
                    f'{len(self.phases)}/{len(self.phases)} done {elapsed}')
        spin = SPINNER[self._spin % len(SPINNER)]
        detail = f' {self.current.detail}' if self.current.detail else ''
        return (f'{self.label} {spin} [{bar}] {100.0 * fraction:3.0f}% '
                f'{self._index_of(self.current.name)}/{len(self.phases)} '
                f'{self.current.name}{detail} {elapsed}')

    def _render(self):
        if not self.enabled or not self.tty:
            return
        self._draw()

    def _draw(self):
        text = self._line()
        pad = max(0, self._drawn - len(text))
        self._write('\r' + text + ' ' * pad)
        self._drawn = len(text)

    def _loop(self):
        while not self._stop.wait(self.tick):
            with self._lock:
                self._spin += 1
                self._render()

    def _write(self, text):
        self.stream.write(text)
        self.stream.flush()

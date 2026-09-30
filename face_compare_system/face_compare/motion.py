"""Small keyed Tk animation scheduler; one timer, no threads or sleeps."""

import time
from dataclasses import dataclass


def ease_out(value):
    return 1-(1-max(0., min(1., value)))**3


def mix_color(start, end, amount):
    amount = max(0., min(1., amount))
    first = [int(start[i:i+2], 16) for i in (1, 3, 5)]
    last = [int(end[i:i+2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(a+(b-a)*amount):02x}" for a, b in zip(first, last))


@dataclass
class _Motion:
    started: float
    duration: float
    paint: object
    repeat: bool


class MotionScheduler:
    """Replacing a key cancels old work; reduced motion resolves immediately."""

    def __init__(self, owner, *, reduced=False, clock=time.monotonic):
        self.owner, self.reduced, self.clock = owner, reduced, clock
        self._motions = {}
        self._timer = None
        self.closed = False

    def start(self, key, duration, paint, *, repeat=False):
        if self.closed:
            return
        self.cancel(key)
        if self.reduced or duration <= 0:
            paint(0. if repeat else 1.)
            return
        self._motions[key] = _Motion(self.clock(), duration, paint, repeat)
        paint(0.)
        self._schedule()

    def cancel(self, key):
        self._motions.pop(key, None)
        if not self._motions:
            self._cancel_timer()

    def _cancel_timer(self):
        if self._timer is not None:
            self.owner.after_cancel(self._timer)
            self._timer = None

    def _schedule(self):
        if self._motions and self._timer is None and not self.closed:
            self._timer = self.owner.after(33, self._tick)

    def _tick(self):
        self._timer = None
        if self.closed:
            return
        now = self.clock()
        for key, motion in list(self._motions.items()):
            if self._motions.get(key) is not motion:
                continue
            elapsed = max(0., (now-motion.started)/motion.duration)
            done = not motion.repeat and elapsed >= 1
            if done:
                self._motions.pop(key)
            motion.paint(elapsed % 1 if motion.repeat else ease_out(elapsed))
        self._schedule()

    def set_reduced(self, reduced):
        self.reduced = bool(reduced)
        if self.reduced:
            self._cancel_timer()
            pending, self._motions = self._motions, {}
            for motion in pending.values():
                motion.paint(0. if motion.repeat else 1.)

    def close(self):
        self.closed = True
        self._cancel_timer()
        self._motions.clear()

from face_compare.motion import MotionScheduler, ease_out, mix_color


class Loop:
    def __init__(self):
        self.now = 0.
        self.next_id = 0
        self.jobs = {}

    def after(self, milliseconds, callback):
        self.next_id += 1
        self.jobs[self.next_id] = callback
        return self.next_id

    def after_cancel(self, key):
        self.jobs.pop(key, None)

    def step(self, now):
        self.now = now
        jobs, self.jobs = self.jobs, {}
        for callback in jobs.values():
            callback()


def scheduler():
    loop = Loop()
    return loop, MotionScheduler(loop, clock=lambda: loop.now)


def test_one_timer_for_many_animations_and_exact_final_values():
    loop, motion = scheduler()
    values = {key: [] for key in range(10)}
    for key in values:
        motion.start(key, 1., values[key].append)
    assert len(loop.jobs) == 1
    loop.step(.5)
    assert all(value[-1] == .875 for value in values.values())
    loop.step(1.)
    assert all(value[-1] == 1. for value in values.values())
    assert not loop.jobs and not motion._motions


def test_replacing_progress_cancels_old_target():
    loop, motion = scheduler()
    old, new = [], []
    motion.start("progress", 1, old.append)
    motion.start("progress", .5, new.append)
    loop.step(.5)
    assert old == [0.]
    assert new == [0., 1.]


def test_reduced_motion_finishes_finite_work_and_stops_repeat():
    loop, motion = scheduler()
    values, spinner = [], []
    motion.start("progress", 1, values.append)
    motion.start("status", 1, spinner.append, repeat=True)
    loop.step(.2)
    motion.set_reduced(True)
    assert values[-1] == 1 and spinner[-1] == 0
    assert not loop.jobs and not motion._motions
    motion.start("next", 1, values.append)
    assert values[-1] == 1 and not loop.jobs


def test_repeat_uses_wall_clock_phase_and_close_cancels_callbacks():
    loop, motion = scheduler()
    values = []
    motion.start("status", 1., values.append, repeat=True)
    loop.step(2.25)
    assert values[-1] == .25
    motion.close()
    loop.step(3.)
    motion.start("ignored", 1., values.append)
    assert values == [0., .25] and not loop.jobs


def test_color_and_easing_clamped():
    assert mix_color("#000000", "#ffffff", -1) == "#000000"
    assert mix_color("#000000", "#ffffff", 1.5) == "#ffffff"
    assert ease_out(-1) == 0 and ease_out(2) == 1

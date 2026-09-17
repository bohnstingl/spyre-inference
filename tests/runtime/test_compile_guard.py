# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The compile guard.

Everything here runs on CPU with ``backend="eager"``: the guard keys on which code
object Dynamo *traces*, which is decided before any backend runs, so a real Spyre
compile would only slow the tests down.

The guard reports through logs and exceptions, so that is what these assert on.
"""

import copy
import threading

import pytest
import torch
from torch._dynamo.utils import counters

from spyre_inference import envs
from spyre_inference.v1.worker import compile_guard
from spyre_inference.v1.worker.compile_guard import (
    CompileGuardLevel,
    CompileKind,
    UnexpectedCompileError,
)


@pytest.fixture(autouse=True)
def isolated_dynamo_state():
    """Give each test its own Dynamo cache so a compile in one is a compile here.

    Without the reset, a second test compiling the same function body hits the
    existing cache entry and never fires the callback.
    """
    saved = copy.deepcopy(counters)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()
    counters.clear()
    counters.update(saved)


@pytest.fixture
def guard():
    """A clean guard, torn down even if the test leaves it armed."""
    compile_guard.reset()
    yield compile_guard
    compile_guard.reset()


@pytest.fixture
def violations(caplog):
    """The guard's warnings, as a list of messages."""

    def read():
        return [r.getMessage() for r in caplog.records if "unexpectedly" in r.getMessage()]

    with caplog.at_level("WARNING"):
        yield read


def _compiled(fn):
    return torch.compile(fn, backend="eager", fullgraph=True, dynamic=False)


class TestLevelParsing:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("off", CompileGuardLevel.OFF),
            ("warn", CompileGuardLevel.WARN),
            ("error", CompileGuardLevel.ERROR),
            ("ERROR", CompileGuardLevel.ERROR),
            ("  warn  ", CompileGuardLevel.WARN),
        ],
    )
    def test_accepts_the_documented_values(self, value, expected):
        assert compile_guard.parse_level(value) is expected

    def test_a_typo_raises_rather_than_silently_disabling_the_guard(self):
        # Silently falling back to "off" would turn a typo in a CI job into a green
        # run that checks nothing. Not a misspelling of a real level: the repo's
        # `typos` pre-commit hook rewrites those, which would invert this assertion.
        with pytest.raises(ValueError, match="Invalid SPYRE_COMPILE_GUARD"):
            compile_guard.parse_level("fatal")

    def test_an_empty_value_is_rejected_too(self):
        with pytest.raises(ValueError, match="Invalid SPYRE_COMPILE_GUARD"):
            compile_guard.parse_level("")

    def test_the_env_default_is_off(self, monkeypatch):
        monkeypatch.delenv("SPYRE_COMPILE_GUARD", raising=False)
        envs.clear_env_cache()
        assert compile_guard.parse_level(envs.SPYRE_COMPILE_GUARD) is CompileGuardLevel.OFF

    def test_the_env_var_selects_the_level(self, monkeypatch):
        monkeypatch.setenv("SPYRE_COMPILE_GUARD", "error")
        envs.clear_env_cache()
        assert compile_guard.parse_level(envs.SPYRE_COMPILE_GUARD) is CompileGuardLevel.ERROR


class TestArming:
    def test_off_installs_nothing(self, guard, violations):
        def fn(x):
            return x * 2

        guard.watch(fn, "fn")
        guard.arm(CompileGuardLevel.OFF)
        assert not guard.is_armed()

        _compiled(fn)(torch.randn(4))

        assert violations() == []

    def test_a_compile_before_arming_is_not_reported(self, guard, violations):
        def fn(x):
            return x * 2

        guard.watch(fn, "fn")
        _compiled(fn)(torch.randn(4))
        guard.arm(CompileGuardLevel.WARN)

        # Compiles the guard was armed *after* must not count against it.
        assert violations() == []

    def test_arming_twice_switches_level_without_double_registering(self, guard):
        def fn(x):
            return x * 2

        guard.watch(fn, "fn")
        guard.arm(CompileGuardLevel.WARN)
        guard.arm(CompileGuardLevel.ERROR)

        assert guard.is_armed()
        assert guard.level() is CompileGuardLevel.ERROR
        # A second callback registration would surface here as a chained raise.
        with pytest.raises(UnexpectedCompileError):
            _compiled(fn)(torch.randn(4))

    def test_disarm_stops_reporting(self, guard, violations):
        def fn(x):
            return x * 2

        guard.watch(fn, "fn")
        guard.arm(CompileGuardLevel.ERROR)
        guard.disarm()

        # Would raise if the callback were still installed.
        _compiled(fn)(torch.randn(4))

        assert not guard.is_armed()
        assert violations() == []

    def test_disarm_without_arm_is_a_noop(self, guard):
        guard.disarm()
        assert not guard.is_armed()

    def test_a_dynamo_reset_does_not_break_disarm(self, guard):
        """``torch._dynamo.reset()`` calls ``callback_handler.clear()``, dropping the
        registration behind the guard's back. Disarming then must not raise."""
        guard.arm(CompileGuardLevel.WARN)
        torch._dynamo.reset()

        guard.disarm()

        assert not guard.is_armed()

    def test_re_arming_after_a_dynamo_reset_reinstalls_the_callback(self, guard):
        """Same clear() as above, but the guard stays 'armed', so re-arming has to
        notice the callback is gone or the guard silently stops reporting."""

        def fn(x):
            return x * 2

        guard.watch(fn, "fn")
        guard.arm(CompileGuardLevel.ERROR)
        torch._dynamo.reset()
        guard.arm(CompileGuardLevel.ERROR)

        with pytest.raises(UnexpectedCompileError):
            _compiled(fn)(torch.randn(4))


class TestClassification:
    def test_a_watched_first_compile_is_reported(self, guard, violations):
        def fn(x):
            return x * 2

        guard.watch(fn, "my kernel")
        guard.arm(CompileGuardLevel.WARN)
        _compiled(fn)(torch.randn(4))

        assert len(violations()) == 1
        assert "my kernel compiled unexpectedly" in violations()[0]

    def test_a_guard_triggered_recompile_is_flagged_as_such(self, guard, violations):
        """The headline case: a shape warmup did not cover, so a guard failed."""

        def fn(x):
            return x * 2

        guard.watch(fn, "my kernel")
        compiled = _compiled(fn)
        compiled(torch.randn(4))
        guard.arm(CompileGuardLevel.WARN)

        compiled(torch.randn(5))

        assert len(violations()) == 1
        assert "my kernel recompiled unexpectedly" in violations()[0]

    def test_an_unwatched_frame_is_named_by_its_code_object(self, guard, violations):
        guard.arm(CompileGuardLevel.WARN)

        _compiled(lambda x: x * 2)(torch.randn(4))

        assert len(violations()) == 1
        assert "<lambda>" in violations()[0]

    def test_the_eager_op_trampoline_is_ignored(self, guard):
        """torch-spyre compiles every eager aten op through one shared torch frame.

        Those compiles continue for the whole run by design, so they must never be
        reported. Exercised through the classifier directly because the real path
        needs a Spyre device.
        """
        kind, label = guard._guard._classify(_fake_eager_op_code())

        assert kind is CompileKind.EAGER_OP
        assert label == "torch-spyre eager op"

    def test_an_unresolvable_frame_is_unknown(self, guard):
        kind, label = guard._guard._classify(None)

        assert kind is CompileKind.UNKNOWN
        assert label == "<unknown frame>"


class TestErrorLevel:
    def test_a_watched_compile_raises(self, guard):
        def fn(x):
            return x * 2

        guard.watch(fn, "block 0")
        guard.arm(CompileGuardLevel.ERROR)

        with pytest.raises(UnexpectedCompileError, match="block 0 compiled unexpectedly"):
            _compiled(fn)(torch.randn(4))

    def test_a_watched_recompile_raises_and_names_the_recompile(self, guard):
        def fn(x):
            return x * 2

        guard.watch(fn, "block 0")
        compiled = _compiled(fn)
        compiled(torch.randn(4))
        guard.arm(CompileGuardLevel.ERROR)

        with pytest.raises(UnexpectedCompileError, match="block 0 recompiled unexpectedly"):
            compiled(torch.randn(5))

    def test_the_error_escapes_dynamo_unwrapped(self, guard):
        """Dynamo rewraps a callback exception as InternalTorchDynamoError unless the
        type is on its passthrough list, which is why this inherits AssertionError.
        Without that, callers could not catch UnexpectedCompileError at all."""

        def fn(x):
            return x * 2

        guard.watch(fn, "block 0")
        guard.arm(CompileGuardLevel.ERROR)

        with pytest.raises(UnexpectedCompileError) as excinfo:
            _compiled(fn)(torch.randn(4))

        assert type(excinfo.value) is UnexpectedCompileError

    def test_the_message_points_at_the_diagnostic_and_the_escape_hatch(self, guard):
        def fn(x):
            return x * 2

        guard.watch(fn, "block 0")
        guard.arm(CompileGuardLevel.ERROR)

        with pytest.raises(UnexpectedCompileError) as excinfo:
            _compiled(fn)(torch.randn(4))

        assert "TORCH_LOGS=recompiles" in str(excinfo.value)
        assert "SPYRE_COMPILE_GUARD=warn" in str(excinfo.value)

    def test_an_unknown_frame_never_raises(self, guard, violations):
        """A torch upgrade that moves the eager-op trampoline would land every eager
        op in UNKNOWN. Killing the engine over that is worse than a noisy log."""
        guard.arm(CompileGuardLevel.ERROR)

        _compiled(lambda x: x * 2)(torch.randn(4))

        assert len(violations()) == 1

    def test_warn_level_does_not_raise(self, guard, violations):
        def fn(x):
            return x * 2

        guard.watch(fn, "block 0")
        guard.arm(CompileGuardLevel.WARN)

        _compiled(fn)(torch.randn(4))

        assert len(violations()) == 1


class TestAllowCompile:
    def test_it_suppresses_reporting(self, guard, violations):
        def fn(x):
            return x * 2

        guard.watch(fn, "recorder kernel")
        guard.arm(CompileGuardLevel.ERROR)

        with guard.allow_compile("graph recorder"):
            _compiled(fn)(torch.randn(4))

        assert violations() == []

    def test_reporting_resumes_afterwards(self, guard):
        def outside(x):
            return x + 1

        guard.watch(outside, "outside")
        guard.arm(CompileGuardLevel.ERROR)

        with guard.allow_compile("recorder"):
            pass

        with pytest.raises(UnexpectedCompileError):
            _compiled(outside)(torch.randn(4))

    def test_it_resumes_even_if_the_body_raises(self, guard):
        def fn(x):
            return x * 2

        guard.watch(fn, "fn")
        guard.arm(CompileGuardLevel.ERROR)

        with pytest.raises(RuntimeError, match="boom"), guard.allow_compile("recorder"):
            raise RuntimeError("boom")

        with pytest.raises(UnexpectedCompileError):
            _compiled(fn)(torch.randn(4))

    def test_nesting_is_balanced(self, guard, violations):
        def fn(x):
            return x * 2

        guard.watch(fn, "fn")
        guard.arm(CompileGuardLevel.ERROR)

        with guard.allow_compile("outer"):
            with guard.allow_compile("inner"):
                pass
            # Still suppressed: the outer block has not exited.
            _compiled(fn)(torch.randn(4))

        assert violations() == []

    def test_suppression_is_per_thread(self, guard, violations):
        """A recorder on one thread must not blind the guard to a compile triggered
        on another."""

        def fn(x):
            return x * 2

        guard.watch(fn, "fn")
        guard.arm(CompileGuardLevel.WARN)

        entered = threading.Event()
        release = threading.Event()

        def hold():
            with guard.allow_compile("recorder"):
                entered.set()
                release.wait(timeout=5)

        thread = threading.Thread(target=hold)
        thread.start()
        assert entered.wait(timeout=5)
        try:
            _compiled(fn)(torch.randn(4))
        finally:
            release.set()
            thread.join(timeout=5)

        assert len(violations()) == 1


class TestWatch:
    def test_a_plain_function_is_registered(self, guard):
        def fn(x):
            return x

        guard.watch(fn, "fn")

        assert fn.__code__ in guard._guard.watched_labels()

    def test_a_module_registers_its_forward(self, guard):
        class Block(torch.nn.Module):
            def forward(self, x):
                return x

        guard.watch(Block(), "block")

        assert Block.forward.__code__ in guard._guard.watched_labels()

    def test_a_compiled_module_registers_the_original_forward(self, guard):
        """Dynamo traces the *original* frame, so registering the wrapper must
        resolve through to the code object the callback will actually see."""

        class Block(torch.nn.Module):
            def forward(self, x):
                return x * 2

        compiled = torch.compile(Block(), backend="eager", dynamic=False)
        guard.watch(compiled, "block")

        assert Block.forward.__code__ in guard._guard.watched_labels()

    def test_a_bound_method_registers_the_underlying_function(self, guard):
        class Layer:
            def kernel(self, x):
                return x

        guard.watch(Layer().kernel, "kernel")

        assert Layer.kernel.__code__ in guard._guard.watched_labels()

    def test_the_first_label_wins_for_a_shared_code_object(self, guard):
        """Identical blocks share one forward code object -- that is why per-block
        compile shares an artifact -- so re-registration must not churn the label."""

        class Block(torch.nn.Module):
            def forward(self, x):
                return x

        guard.watch(Block(), "first")
        guard.watch(Block(), "second")

        assert guard._guard.watched_labels()[Block.forward.__code__] == "first"

    def test_watching_after_arming_takes_effect(self, guard, violations):
        def fn(x):
            return x * 2

        guard.arm(CompileGuardLevel.WARN)
        guard.watch(fn, "late registration")
        _compiled(fn)(torch.randn(4))

        assert "late registration compiled unexpectedly" in violations()[0]


class TestReporting:
    def test_a_repeating_violation_logs_once(self, guard, violations):
        """A recompiling shape usually repeats every step; the point is to name it,
        not to flood the log."""

        def fn(x):
            return x * 2

        guard.watch(fn, "block 0")
        guard.arm(CompileGuardLevel.WARN)
        compiled = _compiled(fn)

        compiled(torch.randn(4))
        compiled(torch.randn(5))
        compiled(torch.randn(6))

        messages = violations()
        # First-compile and recompile are distinct reports; each logs only once.
        assert sum("block 0 compiled" in m for m in messages) == 1
        assert sum("block 0 recompiled" in m for m in messages) == 1


class TestRecompileIdParsing:
    @pytest.mark.parametrize(
        ("compile_id", "expected"),
        [
            ("0/0", False),
            ("1/0", False),
            ("1/1", True),
            ("12/7", True),
            ("2/1/0", True),
            ("-/0", False),
            ("?", False),
            ("", False),
        ],
    )
    def test_a_nonzero_sub_index_means_a_recompile(self, compile_id, expected):
        assert compile_guard._is_recompile(compile_id) is expected


def _fake_eager_op_code():
    """A code object whose filename matches torch-spyre's eager-op trampoline."""
    source = "def inner(*args, **kwargs):\n    return None\n"
    namespace: dict = {}
    path = "/some/prefix/torch/_dynamo/external_utils.py"
    exec(compile(source, path, "exec"), namespace)
    return namespace["inner"].__code__

import logging

import pytest

import verifiers as vf


class TestSetupLogging:
    """Tests for the vf.setup_logging function."""

    def test_setup_logging_sets_level(self):
        """Verify that setup_logging sets the log level."""
        logger = logging.getLogger("verifiers")

        vf.setup_logging("DEBUG")
        assert logger.level == logging.DEBUG

        vf.setup_logging("WARNING")
        assert logger.level == logging.WARNING

    def test_setup_logging_overrides_on_multiple_calls(self):
        """Verify that calling setup_logging multiple times overrides the config."""
        logger = logging.getLogger("verifiers")

        vf.setup_logging("INFO")
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1

        vf.setup_logging("DEBUG")
        assert logger.level == logging.DEBUG
        # Should still have exactly one handler, not two
        assert len(logger.handlers) == 1

    def test_setup_logging_case_insensitive(self):
        """Test that level names are case-insensitive."""
        logger = logging.getLogger("verifiers")

        vf.setup_logging("debug")
        assert logger.level == logging.DEBUG

        vf.setup_logging("Warning")
        assert logger.level == logging.WARNING


class TestLogLevel:
    """Tests for the vf.log_level context manager."""

    def test_log_level_changes_level_within_context(self):
        """Verify that the log level is changed within the context."""
        logger = logging.getLogger("verifiers")
        original_level = logger.level

        with vf.log_level("DEBUG"):
            assert logger.level == logging.DEBUG

        # Should be restored after exiting the context
        assert logger.level == original_level

    def test_log_level_accepts_string(self):
        """Test that log_level accepts string level names."""
        logger = logging.getLogger("verifiers")
        original_level = logger.level

        with vf.log_level("WARNING"):
            assert logger.level == logging.WARNING

        assert logger.level == original_level

    def test_log_level_accepts_int(self):
        """Test that log_level accepts integer level values."""
        logger = logging.getLogger("verifiers")
        original_level = logger.level

        with vf.log_level(logging.ERROR):
            assert logger.level == logging.ERROR

        assert logger.level == original_level

    def test_log_level_restores_on_exception(self):
        """Verify the log level is restored even when an exception occurs."""
        logger = logging.getLogger("verifiers")
        original_level = logger.level

        try:
            with vf.log_level("CRITICAL"):
                assert logger.level == logging.CRITICAL
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still be restored after the exception
        assert logger.level == original_level

    def test_log_level_case_insensitive(self):
        """Test that string level names are case-insensitive."""
        logger = logging.getLogger("verifiers")
        original_level = logger.level

        with vf.log_level("debug"):
            assert logger.level == logging.DEBUG

        with vf.log_level("Debug"):
            assert logger.level == logging.DEBUG

        assert logger.level == original_level


class TestJsonLoggingMutesHttpNoise:
    """Tests that setup_logging(json_logging=True) mutes httpcore/httpx DEBUG."""

    @pytest.fixture
    def restore_logging_state(self):
        """Snapshot + restore root, verifiers, httpcore, and httpx logger state."""
        names = (None, "verifiers", "httpcore", "httpx")
        saved: list[tuple[logging.Logger, int, list[logging.Handler], bool]] = []
        for name in names:
            lg = logging.getLogger(name) if name else logging.getLogger()
            saved.append((lg, lg.level, list(lg.handlers), lg.propagate))
        try:
            yield
        finally:
            for lg, level, handlers, propagate in saved:
                lg.setLevel(level)
                lg.handlers = handlers
                lg.propagate = propagate

    def test_json_logging_debug_mutes_httpcore_and_httpx(self, restore_logging_state):
        """setup_logging at DEBUG with json_logging=True pins httpcore/httpx to WARNING."""
        vf.setup_logging(level="DEBUG", json_logging=True)

        assert logging.getLogger("httpcore").getEffectiveLevel() == logging.WARNING
        assert logging.getLogger("httpx").getEffectiveLevel() == logging.WARNING

        # Root stays at DEBUG so user code and other namespaces still emit DEBUG.
        assert logging.getLogger().getEffectiveLevel() == logging.DEBUG
        # A sibling user logger (no explicit level) inherits root's DEBUG, proving
        # we didn't over-mute.
        assert (
            logging.getLogger("some.random.user.logger").getEffectiveLevel()
            == logging.DEBUG
        )

    def test_non_json_logging_does_not_mute_httpcore(self, restore_logging_state):
        """json_logging=False must not mutate httpcore/httpx levels."""
        # Start from a known baseline.
        logging.getLogger("httpcore").setLevel(logging.NOTSET)
        logging.getLogger("httpx").setLevel(logging.NOTSET)

        vf.setup_logging(level="DEBUG", json_logging=False)

        # Unchanged by the non-json path.
        assert logging.getLogger("httpcore").level == logging.NOTSET
        assert logging.getLogger("httpx").level == logging.NOTSET


class TestQuietVerifiers:
    """Tests for the vf.quiet_verifiers context manager."""

    def test_quiet_verifiers_sets_warning_level(self):
        """Verify that quiet_verifiers sets the log level to WARNING."""
        logger = logging.getLogger("verifiers")
        original_level = logger.level

        with vf.quiet_verifiers():
            assert logger.level == logging.WARNING

        assert logger.level == original_level

    def test_quiet_verifiers_restores_on_exception(self):
        """Verify the log level is restored even when an exception occurs."""
        logger = logging.getLogger("verifiers")
        original_level = logger.level

        try:
            with vf.quiet_verifiers():
                assert logger.level == logging.WARNING
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert logger.level == original_level

import logging

import verifiers as vf


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

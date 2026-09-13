def pytest_addoption(parser):
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run tests that hit a real, already-running Spooky server (see test_spooky_client_live.py)",
    )

import matplotlib
from matplotlib.testing.conftest import *  # noqa


def pytest_configure(config):
    # config is initialized here rather than in pytest.ini so that `pytest
    # --pyargs matplotlib` (which would not find pytest.ini) works.  The only
    # entries in pytest.ini set minversion (which is checked earlier),
    # testpaths/python_files, as they are required to properly find the tests
    for key, value in [
        ("markers", "flaky: (Provided by pytest-rerunfailures.)"),
        ("markers", "timeout: (Provided by pytest-timeout.)"),
        ("markers", "backend: Set alternate Matplotlib backend temporarily."),
        ("markers", "style: Set alternate Matplotlib style temporarily."),
        ("markers", "baseline_images: Compare output against references."),
        ("markers", "pytz: Tests that require pytz to be installed."),
        # ("filterwarnings", "error"),
    ]:
        config.addinivalue_line(key, value)

    matplotlib.use("agg", force=True)
    matplotlib._called_from_pytest = True
    matplotlib._init_tests()

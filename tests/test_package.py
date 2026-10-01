import re

import lunarlander_rl


def test_version_is_pep440() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+|a\d+|b\d+|rc\d+)?", lunarlander_rl.__version__)

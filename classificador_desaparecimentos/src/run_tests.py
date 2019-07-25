import os
import sys
import pytest
import glob

IS_TRAVIS = os.environ.get('IS_TRAVIS', '')

if not IS_TRAVIS:
    types = ['whl', 'egg']
    filenames = []
    for filetype in types:
        filenames.extend(
            glob.glob("packages/*.{}".format(filetype)))

    abspath = os.path.dirname(os.path.abspath(__file__))

    for filename in filenames:
        sys.path.insert(
            0,
            abspath + "/../" + filename)

pytest.main(['--cov=src', '--cov-report', 'term-missing'])

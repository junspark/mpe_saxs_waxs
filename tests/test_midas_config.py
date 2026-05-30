#!/usr/bin/env python3

import os
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MIDAS_HOME = SCRIPT_DIR.parent
sys.path.insert(0, str(MIDAS_HOME / "utils"))

import midas_config


class TestMidasConfig(unittest.TestCase):
    def setUp(self):
        self._original_env = os.environ.get("HDF5_PLUGIN_PATH")

    def tearDown(self):
        if self._original_env is None:
            os.environ.pop("HDF5_PLUGIN_PATH", None)
        else:
            os.environ["HDF5_PLUGIN_PATH"] = self._original_env

    def test_unset_plugin_path_when_all_paths_invalid(self):
        os.environ["HDF5_PLUGIN_PATH"] = "/this/path/does/not/exist"
        midas_config.sanitize_hdf5_plugin_path()
        self.assertNotIn("HDF5_PLUGIN_PATH", os.environ)

    def test_keep_only_valid_plugin_paths(self):
        with tempfile.TemporaryDirectory() as valid_dir:
            os.environ["HDF5_PLUGIN_PATH"] = (
                f"/invalid/path{os.pathsep}{valid_dir}{os.pathsep}"
            )
            midas_config.sanitize_hdf5_plugin_path()
            self.assertEqual(os.environ["HDF5_PLUGIN_PATH"], valid_dir)


if __name__ == "__main__":
    unittest.main()

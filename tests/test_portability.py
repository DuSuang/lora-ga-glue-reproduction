import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from glue.system_io import file_lock


class LockTests(unittest.TestCase):
    def test_process_exclusion_and_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            same=Path(temp)/"seed-0"/".running.lock"
            other=Path(temp)/"seed-1"/".running.lock"
            program=("import sys; from pathlib import Path; from glue.system_io import file_lock\n"
                     "with file_lock(Path(sys.argv[1])):\n"
                     " print('locked',flush=True); sys.stdin.readline(); sys.exit(0)")
            with subprocess.Popen([sys.executable,"-S","-c",program,str(same)],cwd=Path(__file__).resolve().parents[1],
                                  stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True) as child:
                try:
                    self.assertEqual(child.stdout.readline().strip(),"locked")
                    with self.assertRaises(OSError),file_lock(same):
                        self.fail("two writers acquired the same config")
                    with file_lock(other):
                        self.assertTrue(other.is_file())
                finally:
                    child.communicate("\n",timeout=10)
            with file_lock(same):
                self.assertTrue(same.is_file())

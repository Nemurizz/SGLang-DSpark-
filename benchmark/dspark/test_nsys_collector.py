import sqlite3
import tempfile
from pathlib import Path
import unittest

from audit_nsys_collector import audit


class CollectorTests(unittest.TestCase):
    def database(self, path, *, count=16, other_pid=False, duplicate=False, unsupported=False):
        c = sqlite3.connect(path)
        c.executescript('''create table DIAGNOSTIC_EVENT(severity integer,text text);
            create table CUPTI_ACTIVITY_KIND_KERNEL(correlationId integer,globalPid integer);
            create table CUPTI_ACTIVITY_KIND_RUNTIME(correlationId integer,globalTid integer);
            create table StringIds(id integer,value text);
            create table NVTX_EVENTS(text text,textId integer,end integer);''')
        pid = 1 << 24
        c.executemany('insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?)', [(i,pid) for i in range(count)])
        launch_pid = 2 << 24 if other_pid else pid
        keys = [(i,launch_pid + 7) for i in range(16)]
        c.executemany('insert into CUPTI_ACTIVITY_KIND_RUNTIME values(?,?)', keys * (2 if duplicate else 1))
        c.executemany('insert into NVTX_EVENTS values(?,null,1)', [('nsys_smoke_mm_' + str(i),) for i in range(16)])
        c.execute('insert into DIAGNOSTIC_EVENT values(1,?)', ('Loaded CUPTI library: /tools/libcupti.so.13.2',))
        c.execute('insert into DIAGNOSTIC_EVENT values(2,?)', ('Not all CUDA events might have been collected.',))
        if unsupported:
            c.execute('insert into DIAGNOSTIC_EVENT values(2,?)', ('Driver is not supported by this build',))
        c.commit()
        c.close()

    def exercise(self, expected_pass, **options):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'trace.sqlite'
            self.database(path, **options)
            if expected_pass:
                result = audit(path, smoke=True)
                self.assertTrue(result['bounded_smoke_coverage_passed'])
                self.assertEqual(result['runtime_correlated_kernels'], 16)
                self.assertTrue(result['potential_missing_events_messages'])
            else:
                with self.assertRaises(ValueError):
                    audit(path, smoke=True)

    def test_smoke_and_diagnostics_preserved(self):
        self.exercise(True)

    def test_duplicate_launch_key_not_double_counted(self):
        self.exercise(True, duplicate=True)

    def test_other_process_not_false_correlated(self):
        self.exercise(False, other_pid=True)

    def test_missing_kernel_rejected(self):
        self.exercise(False, count=15)

    def test_unsupported_driver_rejected(self):
        self.exercise(False, unsupported=True)


if __name__ == '__main__':
    unittest.main()

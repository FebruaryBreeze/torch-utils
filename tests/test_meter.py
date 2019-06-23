import time
import unittest

import torch_utils


class MyTestCase(unittest.TestCase):
    def test_average_meter(self):
        average = torch_utils.AverageMeter('Test', 4)
        average.update(1)
        average.update(2)
        self.assertAlmostEqual(average.val, 2)
        self.assertAlmostEqual(average.avg, 1.5)

        average.update(3)
        average.update(4)
        self.assertAlmostEqual(average.val, 4)
        self.assertAlmostEqual(average.avg, 2.5)

        average.update(5)
        average.update(6)
        self.assertAlmostEqual(average.val, 6)
        self.assertAlmostEqual(average.avg, 4.5)

        self.assertEqual(str(average), 'Test 6.00 (4.50)')

        average.reset()
        self.assertIsNone(average.val)

    def test_time_meter(self):
        elapsed = torch_utils.TimeMeter('Test', 4)
        with elapsed:
            time.sleep(0.2)
        self.assertAlmostEqual(elapsed.val, 0.2, delta=0.05)

    def test_weighted_meter(self):
        weighted = torch_utils.WeightedMeter('Test')
        weighted.update(1)
        weighted.update(2)
        self.assertAlmostEqual(weighted.val, 2)
        self.assertAlmostEqual(weighted.avg, 1.5)

        weighted.update(3)
        weighted.update(4)
        self.assertAlmostEqual(weighted.val, 4)
        self.assertAlmostEqual(weighted.avg, 2.5)

        weighted.update(5)
        weighted.update(6)
        self.assertAlmostEqual(weighted.val, 6)
        self.assertAlmostEqual(weighted.avg, 3.5)
        self.assertAlmostEqual(weighted.count, 6)

        self.assertEqual(str(weighted), 'Test 6.00 (3.50)')

        weighted.reset(sum=9.78, count=3)
        self.assertEqual(str(weighted), 'Test 3.26 (3.26)')

    def test_progress_meter(self):
        progress = torch_utils.ProgressMeter(total_steps=100, total_epochs=10)
        progress.update(step=10)
        self.assertAlmostEqual(progress.ratio, 0.1)
        self.assertEqual(str(progress), 'Step 10/100=10.0% (1/10)')
        self.assertFalse(progress.is_last_step)


if __name__ == '__main__':
    unittest.main()

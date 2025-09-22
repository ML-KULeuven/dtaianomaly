import numpy as np

from dtaianomaly.anomaly_detection import ForecastDataset, ReconstructionDataset


class TestForecastDataset:

    def test(self):
        dataset = ForecastDataset(np.arange(17), 5, 1, False, "cpu", 1)
        assert len(dataset) == 17 - 5 - 1 + 1
        for i in range(12):
            history, future = dataset[i]
            assert np.array_equal(history, np.arange(5) + i)
            assert np.array_equal(future, np.arange(1) + 5 + i)

    def test_multivariate(self):
        dataset = ForecastDataset(
            np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]).T, 2, 1, False, "cpu", 1
        )
        assert len(dataset) == 4
        assert np.array_equal(dataset[0][0], [[1, 1], [2, 2]])
        assert np.array_equal(dataset[0][1], [[3, 3]])
        assert np.array_equal(dataset[1][0], [[2, 2], [3, 3]])
        assert np.array_equal(dataset[1][1], [[4, 4]])
        assert np.array_equal(dataset[2][0], [[3, 3], [4, 4]])
        assert np.array_equal(dataset[2][1], [[5, 5]])
        assert np.array_equal(dataset[3][0], [[4, 4], [5, 5]])
        assert np.array_equal(dataset[3][1], [[6, 6]])

    def test_stride_2(self):
        dataset = ForecastDataset(np.arange(17), 5, 2, False, "cpu", 1)
        """
        0 4 5
        2 6 7
        4 8 9
        6 10 11
        8 12 13
        10 14 15
        12 16 17 -> 11 15 16
        """
        assert len(dataset) == 7

        history, future = dataset[0]
        assert np.array_equal(history, np.arange(5))
        assert np.array_equal(future, np.arange(1) + 5)

        history, future = dataset[1]
        assert np.array_equal(history, np.arange(5) + 2)
        assert np.array_equal(future, np.arange(1) + 5 + 2)

        history, future = dataset[2]
        assert np.array_equal(history, np.arange(5) + 4)
        assert np.array_equal(future, np.arange(1) + 5 + 4)

        history, future = dataset[3]
        assert np.array_equal(history, np.arange(5) + 6)
        assert np.array_equal(future, np.arange(1) + 5 + 6)

        history, future = dataset[4]
        assert np.array_equal(history, np.arange(5) + 8)
        assert np.array_equal(future, np.arange(1) + 5 + 8)

        history, future = dataset[5]
        assert np.array_equal(history, np.arange(5) + 10)
        assert np.array_equal(future, np.arange(1) + 5 + 10)

        history, future = dataset[6]
        assert np.array_equal(history, np.arange(5) + 11)
        assert np.array_equal(future, np.arange(1) + 5 + 11)

    def test_stride_3(self):
        dataset = ForecastDataset(np.arange(17), 5, 3, False, "cpu", 1)
        """
        0 4 5
        3 7 8
        6 10 11
        9 13 14
        12 16 17 -> 11 15 16
        """
        assert len(dataset) == 5

        history, future = dataset[0]
        assert np.array_equal(history, np.arange(5))
        assert np.array_equal(future, np.arange(1) + 5)

        history, future = dataset[1]
        assert np.array_equal(history, np.arange(5) + 3)
        assert np.array_equal(future, np.arange(1) + 5 + 3)

        history, future = dataset[2]
        assert np.array_equal(history, np.arange(5) + 6)
        assert np.array_equal(future, np.arange(1) + 5 + 6)

        history, future = dataset[3]
        assert np.array_equal(history, np.arange(5) + 9)
        assert np.array_equal(future, np.arange(1) + 5 + 9)

        history, future = dataset[4]
        assert np.array_equal(history, np.arange(5) + 11)
        assert np.array_equal(future, np.arange(1) + 5 + 11)

    def test_scaling(self):
        dataset = ForecastDataset(
            np.array([5, 10, 15, 9, 15, 10]), 2, 3, True, "cpu", 1
        )
        assert np.allclose(dataset[0][0], np.array([-1.22474487, 0]))
        assert np.allclose(dataset[0][1], np.array([1.22474487]))
        assert np.allclose(dataset[1][0], np.array([-0.88900089, 1.3970014]))
        assert np.allclose(dataset[1][1], np.array([-0.50800051]))

        dataset = ForecastDataset(
            np.array([5, 10, 15, 9, 15, 10]), 2, 3, False, "cpu", 1
        )
        assert np.array_equal(dataset[0][0], np.array([5, 10]))
        assert np.array_equal(dataset[0][1], np.array([15]))
        assert np.array_equal(dataset[1][0], np.array([9, 15]))
        assert np.array_equal(dataset[1][1], np.array([10]))

    def test_scaling_multivariate(self):
        dataset = ForecastDataset(
            np.array([[5, 10, 15, 5, 10, 15], [9, 15, 10, 9, 15, 10]]).T,
            2,
            3,
            True,
            "cpu",
            1,
        )
        assert len(dataset) == 2
        for i in range(2):
            assert np.allclose(
                dataset[i][0], [[-1.22474487, -0.88900089], [0, 1.3970014]]
            )
            assert np.allclose(dataset[i][1], [[1.22474487, -0.50800051]])

    def test_forecast_length(self):
        dataset = ForecastDataset(np.arange(17), 5, 1, False, "cpu", 3)
        assert len(dataset) == 10
        for i in range(10):
            history, future = dataset[i]
            assert np.array_equal(history, np.arange(5) + i)
            assert np.array_equal(future, np.arange(3) + 5 + i)


class TestReconstructionDataset:

    def test(self):
        dataset = ReconstructionDataset(
            np.arange(17),
            5,
            1,
            False,
            "cpu",
        )
        assert len(dataset) == 13
        for i in range(13):
            assert np.array_equal(dataset[i][0], np.arange(5) + i)

    def test_multivariate(self):
        dataset = ReconstructionDataset(
            np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]).T,
            3,
            1,
            False,
            "cpu",
        )
        assert len(dataset) == 4
        assert np.array_equal(dataset[0][0], [[1, 1], [2, 2], [3, 3]])
        assert np.array_equal(dataset[1][0], [[2, 2], [3, 3], [4, 4]])
        assert np.array_equal(dataset[2][0], [[3, 3], [4, 4], [5, 5]])
        assert np.array_equal(dataset[3][0], [[4, 4], [5, 5], [6, 6]])

    def test_stride_2(self):
        dataset = ReconstructionDataset(
            np.arange(17),
            6,
            2,
            False,
            "cpu",
        )
        """
        0 5
        2 7
        4 9
        6 11
        8 13
        10 15
        12 17 -> 11 16
        """
        assert len(dataset) == 7
        assert np.array_equal(dataset[0][0], np.arange(6))
        assert np.array_equal(dataset[1][0], np.arange(6) + 2)
        assert np.array_equal(dataset[2][0], np.arange(6) + 4)
        assert np.array_equal(dataset[3][0], np.arange(6) + 6)
        assert np.array_equal(dataset[4][0], np.arange(6) + 8)
        assert np.array_equal(dataset[5][0], np.arange(6) + 10)
        assert np.array_equal(dataset[6][0], np.arange(6) + 11)

    def test_stride_3(self):
        dataset = ReconstructionDataset(
            np.arange(17),
            6,
            3,
            False,
            "cpu",
        )
        """
        0 5
        3 8
        6 11
        9 14
        12 17 -> 11 16
        """
        assert len(dataset) == 5
        assert np.array_equal(dataset[0][0], np.arange(6))
        assert np.array_equal(dataset[1][0], np.arange(6) + 3)
        assert np.array_equal(dataset[2][0], np.arange(6) + 6)
        assert np.array_equal(dataset[3][0], np.arange(6) + 9)
        assert np.array_equal(dataset[4][0], np.arange(6) + 11)

    def test_scaling(self):
        dataset = ReconstructionDataset(
            np.array([5, 10, 15, 9, 15, 10]),
            3,
            3,
            False,
            "cpu",
        )
        assert np.allclose(dataset[0], np.array([5, 10, 15]))
        assert np.allclose(dataset[1], np.array([9, 15, 10]))

        dataset = ReconstructionDataset(
            np.array([5, 10, 15, 9, 15, 10]),
            3,
            3,
            True,
            "cpu",
        )
        assert np.allclose(dataset[0][0], np.array([-1.22474487, 0, 1.22474487]))
        assert np.allclose(
            dataset[1][0], np.array([-0.88900089, 1.3970014, -0.50800051])
        )

    def test_scaling_multivariate(self):
        dataset = ReconstructionDataset(
            np.array([[5, 10, 15, 5, 10, 15], [9, 15, 10, 9, 15, 10]]).T,
            3,
            3,
            True,
            "cpu",
        )
        assert len(dataset) == 2
        for i in range(2):
            assert np.allclose(
                dataset[i][0],
                [[-1.22474487, -0.88900089], [0, 1.3970014], [1.22474487, -0.50800051]],
            )

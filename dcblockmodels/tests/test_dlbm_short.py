import pytest
from .test_dlbm import TestDLBM


class TestShort(TestDLBM):

    @pytest.mark.parametrize('test_data', [TestDLBM.test_data_setups[1]], indirect=True)
    @pytest.mark.parametrize('sparse_X', [False, True])
    def test_sparse(self, test_data, sparse_X):
        model, Z, W, level = self.fit_model(
            test_data=test_data,
            sparse_X=sparse_X,
            em_type='VEM'
        )
        self.assert_metrics(model, Z, W, level, TestDLBM.n_first)

    @pytest.mark.parametrize('test_data', [TestDLBM.test_data_setups[1]], indirect=True)
    @pytest.mark.parametrize('em_type', ['VEM', 'CEM'])
    def test_em_type(self, test_data, em_type):
        model, Z, W, level = self.fit_model(
            test_data=test_data,
            sparse_X=False,
            em_type=em_type
        )
        self.assert_metrics(model, Z, W, level, TestDLBM.n_first)

    @pytest.mark.parametrize('test_data', [TestDLBM.test_data_setups[4]], indirect=True)
    def test_absent(self, test_data):
        model, Z, W, level = self.fit_model(
            test_data=test_data,
            sparse_X=False,
            em_type='VEM'
        )
        self.assert_metrics(model, Z, W, level, TestDLBM.n_first)

    @pytest.mark.parametrize('test_data', [TestDLBM.test_data_setups[1]], indirect=True)
    @pytest.mark.parametrize('dtype', ['float32', 'float64'])
    def test_dtype(self, test_data, dtype):
        model, Z, W, level = self.fit_model(
            test_data=test_data,
            sparse_X=False,
            em_type='VEM',
            dtype=dtype
        )
        self.assert_metrics(model, Z, W, level, TestDLBM.n_first)

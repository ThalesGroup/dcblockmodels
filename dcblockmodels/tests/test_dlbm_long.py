import pytest
from .test_dlbm import TestDLBM


class TestLong(TestDLBM):

    @pytest.mark.parametrize('test_data', TestDLBM.test_data_setups, indirect=True)
    @pytest.mark.parametrize('sparse_X', [False, True])
    @pytest.mark.parametrize('em_type', ['VEM', 'CEM'])
    def test_fitted_model(
            self,
            test_data,
            sparse_X,
            em_type):

        model, Z, W, level = self.fit_model(
            test_data,
            sparse_X,
            em_type,
        )
        self.assert_metrics(model, Z, W, level, TestDLBM.n_first)

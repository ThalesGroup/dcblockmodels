import pytest

from .test_hlbm import TestHLBM


class TestLong(TestHLBM):

    @pytest.mark.parametrize('test_data', TestHLBM.test_data_setups, indirect=True)
    @pytest.mark.parametrize('sparse_X', [False, True])
    @pytest.mark.parametrize('estimated_margins', [False, True])
    @pytest.mark.parametrize('regularize', [False, True])
    @pytest.mark.parametrize('em_type', ['VEM', 'CEM'])
    @pytest.mark.parametrize('multiplicative_init', [False, True])
    def test_fitted_model(
            self,
            test_data,
            sparse_X,
            estimated_margins,
            regularize,
            em_type,
            multiplicative_init):

        model, Z, W, level = self.fit_model(
            test_data,
            sparse_X,
            estimated_margins,
            regularize,
            em_type,
            multiplicative_init
        )
        self.assert_metrics(model, Z, W, level, TestHLBM.n_first)

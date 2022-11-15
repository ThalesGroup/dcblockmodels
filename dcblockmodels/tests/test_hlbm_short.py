import warnings
import pytest

from .test_hlbm import TestHLBM

warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestShort(TestHLBM):

    @pytest.mark.parametrize('test_data', [TestHLBM.test_data_setups[1]], indirect=True)
    @pytest.mark.parametrize('sparse_X', [False, True])
    def test_sparse(self, test_data, sparse_X):
        model, Z, W, level = self.fit_model(
            test_data=test_data,
            sparse_X=sparse_X,
            estimated_margins=True,
            regularize=False,
            em_type='VEM',
            multiplicative_init=False
        )
        self.assert_metrics(model, Z, W, level, TestHLBM.n_first)

    @pytest.mark.parametrize('test_data', [TestHLBM.test_data_setups[1]], indirect=True)
    @pytest.mark.parametrize('estimated_margins', [False, True])
    def test_estimated_margins(self, test_data, estimated_margins):
        model, Z, W, level = self.fit_model(
            test_data=test_data,
            sparse_X=False,
            estimated_margins=estimated_margins,
            regularize=False,
            em_type='VEM',
            multiplicative_init=False
        )
        self.assert_metrics(model, Z, W, level, TestHLBM.n_first)

    @pytest.mark.parametrize('test_data', [TestHLBM.test_data_setups[1]], indirect=True)
    @pytest.mark.parametrize('regularize', [False, True])
    def test_regularize(self, test_data, regularize):
        model, Z, W, level = self.fit_model(
            test_data=test_data,
            sparse_X=False,
            estimated_margins=True,
            regularize=regularize,
            em_type='VEM',
            multiplicative_init=False
        )
        self.assert_metrics(model, Z, W, level, TestHLBM.n_first)

    @pytest.mark.parametrize('test_data', [TestHLBM.test_data_setups[1]], indirect=True)
    @pytest.mark.parametrize('em_type', ['VEM', 'CEM'])
    def test_em_type(self, test_data, em_type):
        model, Z, W, level = self.fit_model(
            test_data=test_data,
            sparse_X=False,
            estimated_margins=True,
            regularize=False,
            em_type=em_type,
            multiplicative_init=False
        )
        self.assert_metrics(model, Z, W, level, TestHLBM.n_first)

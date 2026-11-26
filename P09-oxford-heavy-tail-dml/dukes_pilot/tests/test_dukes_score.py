"""Independent small-array checks; no model training or Monte Carlo."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import chi2
import dukes_score as ds


def direct_nw(x, y, h, loo=False):
    output = []
    for i in range(len(x)):
        numerator, denominator = 0., 0.
        for j in range(len(x)):
            if loo and i == j:
                continue
            w = np.exp(-0.5*((x[i]-x[j])/h)**2)
            numerator += w*y[j]
            denominator += w
        output.append(numerator/denominator)
    return np.asarray(output)


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([-1.2,-0.4,0.15,0.9,1.4,2.1])
        self.y = np.array([0.1,1.8,-0.5,1.3,0.9,2.4])

    def test_fixed_bandwidth_against_direct_formula(self):
        for loo in (False, True):
            assert_allclose(ds.nw_predict(self.x,self.y,0.7,loo),
                            direct_nw(self.x,self.y,0.7,loo),rtol=2e-14,atol=2e-14)
        self.assertGreater(np.max(np.abs(ds.nw_predict(self.x,self.y,.7,True)-
                                        ds.nw_predict(self.x,self.y,.7,False))), .01)

    def test_grid_objective_is_loo_and_final_includes_self(self):
        cfg = {"lower_multiplier":.5,"upper_multiplier":2.,"grid_size":9,"refine":False}
        fitted, meta = ds.select_nw(self.x,self.y,cfg)
        scale = np.std(self.x,ddof=1)*len(self.x)**(-.2)
        grid = np.exp(np.linspace(np.log(.5*scale),np.log(2*scale),9))
        losses = [np.mean((self.y-direct_nw(self.x,self.y,h,True))**2) for h in grid]
        best = np.argmin(losses)
        self.assertAlmostEqual(meta["bandwidth"],grid[best],places=14)
        self.assertAlmostEqual(meta["cv_loss"],losses[best],places=14)
        assert_allclose(fitted,direct_nw(self.x,self.y,grid[best]),rtol=2e-14,atol=2e-14)

    def test_default_bandwidth_deterministic(self):
        first, d1 = ds.select_nw(self.x,self.y)
        second, d2 = ds.select_nw(self.x,self.y)
        assert_array_equal(first,second)
        self.assertEqual(d1,d2)

    def test_constant_index_is_explicit_extension(self):
        for constant in (0.,2.5):
            fitted, diag = ds.select_nw(np.full(6,constant),self.y)
            assert_allclose(fitted,np.mean(self.y))
            self.assertTrue(diag["constant_index"])
            self.assertIsNone(diag["bandwidth"])
            loo = (np.sum(self.y)-self.y)/5
            self.assertAlmostEqual(diag["cv_loss"],np.mean((self.y-loo)**2))

    def test_zero_directions_reduce_to_standard(self):
        out = ds.corrected_score(self.x,self.y,np.zeros(6),np.zeros(6))
        assert_array_equal(out["scores"],self.x*self.y)
        self.assertEqual(out["diagnostics"]["alpha"],0.)
        self.assertEqual(out["diagnostics"]["beta"],0.)

    def test_updates_and_full_score_independent_formula(self):
        G = direct_nw(self.x,self.y,.7)
        M = direct_nw(self.y,self.x,.9)
        alpha = np.sum(G*self.x)/np.sum(G*G)
        beta = np.sum(M*self.y)/np.sum(M*M)
        rD, rY = self.x-alpha*G,self.y-beta*M
        expected = rD*rY-G*rD-M*rY
        out = ds.corrected_score(self.x,self.y,G,M)
        assert_allclose(out["scores"],expected,rtol=2e-14,atol=2e-14)
        assert_allclose([np.mean(G*out["rD"]),np.mean(M*out["rY"])],0,atol=1e-14)
        self.assertGreater(abs(np.var(expected,ddof=1)-np.var(rD*rY,ddof=1)),.01)
        assert_allclose(np.mean(expected),np.mean(rD*rY),atol=1e-14)

    def test_near_zero_direction_stable_projection(self):
        tiny = self.x*1e-200
        update, meta = ds.stable_projection(tiny,self.y)
        assert_allclose(update,self.x*np.dot(self.x,self.y)/np.dot(self.x,self.x),atol=1e-14)
        self.assertTrue(np.isfinite(meta["coefficient"]))
        self.assertEqual(meta["direction_sum_squares"],0.)
        self.assertFalse(meta["zero_direction"])

    def test_score_standardization_matches_reference_chisquare(self):
        scores = self.x*self.y
        out = ds.score_test(scores)
        v = sum((scores-np.mean(scores))**2)/(len(scores)-1)
        T = np.sqrt(len(scores))*np.mean(scores)/np.sqrt(v)
        self.assertAlmostEqual(out["T"],T)
        self.assertAlmostEqual(out["p_value"],chi2.sf(T*T,1))

    def test_failures_are_exceptions_not_acceptance(self):
        cases = [lambda:ds.score_test(np.ones(5)),lambda:ds.score_test([0,np.nan,1]),
                 lambda:ds.nw_predict(self.x,self.y,0),
                 lambda:ds.select_nw([0,np.inf],[0,1]),
                 lambda:ds.validate_folds([(np.arange(3),np.arange(3))],6)]
        for action in cases:
            with self.assertRaises(ds.NumericalFailure):
                action()

    def test_initial_fits_use_only_training_folds_and_transformed_y(self):
        X = np.arange(6.)[:,None]
        D = self.x
        Y = self.y
        folds = [(np.arange(3,6),np.arange(3)),(np.arange(3),np.arange(3,6))]
        calls = []
        class FakeLearner:
            def __init__(self,seed):
                self.seed = seed
            def fit(self, xx, yy):
                self.train = xx[:,0].astype(int)
                calls.append((self.seed,self.train.copy(),yy.copy()))
                self.mean = np.mean(yy)
                return self
            def predict(self,xx):
                assert not set(xx[:,0].astype(int)) & set(self.train)
                return np.full(len(xx),self.mean)
        with patch.object(ds,"_learner",FakeLearner):
            predictions = ds.crossfit_initial(X,D,Y,.37,folds,900)
        self.assertEqual([c[0] for c in calls],[901,902,1001,1002])
        for k,(train,test) in enumerate(folds):
            assert_array_equal(calls[2*k][1],train)
            assert_array_equal(calls[2*k+1][2],(Y-.37*D)[train])
            assert_allclose(predictions["m_hat"][test],np.mean((Y-.37*D)[train]))
        calls.clear()
        supplied = np.arange(6.)/10
        with patch.object(ds,"_learner",FakeLearner):
            reused = ds.crossfit_initial(X,D,Y,.37,folds,900,supplied)
        assert_array_equal(reused["g_hat"],supplied)
        self.assertEqual(len(calls),2)

    def test_auxiliaries_fold_local_and_restored_order(self):
        folds = [(np.array([0,2,4]),np.array([1,3,5])),(np.array([1,3,5]),np.array([0,2,4]))]
        D,Y,g,m = self.x,self.y,self.x/3,self.y/4
        out = ds.dukes_from_initial(D,Y,.37,g,m,folds)
        for k,(_,test) in enumerate(folds):
            a,b = (D-g)[test],(Y-.37*D-m)[test]
            G,_ = ds.select_nw(g[test],b)
            M,_ = ds.select_nw(m[test],a)
            expected = ds.corrected_score(a,b,G,M)
            assert_array_equal(out["arrays"]["G"][test],G)
            assert_array_equal(out["arrays"]["M"][test],M)
            assert_array_equal(out["scores"][test],expected["scores"])
        assert_array_equal(out["standard_scores"],(D-g)*(Y-.37*D-m))


if __name__ == "__main__":
    unittest.main()

"""Deterministic DGP, benchmark, and estimator equation checks."""
import unittest
import dgp
import numpy as np
from scipy.stats import rankdata, skew, norm
from scipy.special import eval_legendre
import population
import legacy_core as legacy
import methods as common
from estimators import adaptive, interval, rank_values, regular_rank, sandwich_meat
from run import initialize_worker

initialize_worker()

class DesignTests(unittest.TestCase):
    def test_d1_structure(self):
        data=dgp.generate("smoke","D1",50000,402)
        x,v,e=data["x"],data["v"],data["error"]
        m,ell,_=legacy.truth(x,1,"scalar")
        np.testing.assert_allclose(data["d"]-m,v,atol=1e-14)
        np.testing.assert_allclose(data["y"]-ell[:,0]-v,e,atol=1e-14)
        side=x[:,0]<0
        for subset in (side,~side):
            self.assertLess(abs(np.corrcoef(v[subset],e[subset])[0,1]),.03)
            self.assertLess(abs(np.var(v[subset])-1.),.04)
        self.assertAlmostEqual(np.var(e),1.,delta=.04)
        self.assertAlmostEqual(np.var(e[~side])/np.var(e[side]),16,delta=1.)

    def test_d2_mirror(self):
        data=dgp.generate("smoke","D2",50000,402)
        plus=data["error"][data["v"]==1]
        minus=data["error"][data["v"]==-1]
        self.assertTrue((plus>=-1).all() and (minus<=1).all())
        self.assertLess(abs(plus.mean()),.03)
        self.assertLess(abs(minus.mean()),.03)
        self.assertGreater(skew(plus),1.7)
        self.assertLess(skew(minus),-1.7)
        analytic=2*(.25+1.5*np.exp(-2))-1
        self.assertAlmostEqual(2*population.score(0,"D2")[0],analytic,places=11)
        p=population.benchmarks()
        self.assertAlmostEqual(p["D2"]["targets"]["Rank-W"],p["D2"]["rank_score_roots"]["Rank-W"]["reranked"]["root"],places=11)
        for sc in ("D2","D3"):
            self.assertLess(p[sc]["quadrature_max_difference"],1e-7)

    def test_d3_parity(self):
        data=dgp.generate("smoke","D3",50000,402)
        for sign,sigma in ((1,2.),(-1,.5)):
            z=data["error"][data["v"]==sign]/sigma
            self.assertLess(abs(z.mean()),.03)
            self.assertLess(abs(skew(z)),.05)
            self.assertAlmostEqual(np.std(z),1.,delta=.03)
        u=np.linspace(.03,.97,31)
        for l in range(1,7):
            np.testing.assert_allclose(eval_legendre(l,1-2*u),(-1)**l*eval_legendre(l,2*u-1),atol=1e-14)
        m=population.score(0,"D3")
        np.testing.assert_allclose(m[[0,1,2,4,6]],0.,atol=1e-11)
        self.assertGreater(abs(m[3]),.3)
        np.testing.assert_allclose(population.slopes(0,"D3",384)[[3,5,7]],0.,atol=1e-8)
        result=population.benchmarks()["D3"]
        self.assertGreater(result["rank_score_roots"]["Fixed mixed"]["reranked"]["root"],.5)
        self.assertAlmostEqual(result["targets"]["Rank-Ad"],0,places=8)

    def test_named_seeds(self):
        keys={tuple(dgp.seed_key(mode,sc,n,rep,stream)) for mode in dgp.MODES
              for sc in dgp.SCENARIOS for n in dgp.SIZES for rep in range(5) for stream in dgp.STREAMS}
        self.assertEqual(len(keys),3*3*3*5*len(dgp.STREAMS))

class EstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data=dgp.generate("smoke","D3",500,17)
        cls.w,cls.z,cls.f=data["d"],data["y"],data["folds"]
        cls.pre=legacy.direct_w(cls.w,cls.z,cls.f)

    def test_pairwise_independent_w_minimum(self):
        pair=np.concatenate([(self.z[self.f==k][self.w[self.f==k]>0,None]-
                              self.z[self.f==k][self.w[self.f==k]<0][None,:]).ravel()/2
                             for k in range(5)])
        ordered=np.sort(pair)
        lo,hi=ordered[(len(ordered)-1)//2],ordered[len(ordered)//2]
        self.assertGreaterEqual(self.pre,lo-1e-12)
        self.assertLessEqual(self.pre,hi+1e-12)
        objective=lambda tilt: np.mean(abs(pair-tilt))
        self.assertAlmostEqual(objective(self.pre),objective((lo+hi)/2),places=12)

    def test_sandwich_and_single_rank_reference(self):
        old=common.rank_methods(self.w,self.z,self.f,self.pre,np.random.default_rng(18))
        f,w,z=self.f,self.w,self.z
        for name in ("Rank-W","Rank-N"):
            point,ci,meta=regular_rank(name,w,z,f,self.pre,0,True)
            self.assertAlmostEqual(point,old[name][0],places=11)
            self.assertAlmostEqual(ci["naive"]["se"],old[name][1],places=11)
            score=np.zeros(len(f))
            for k in range(5):
                mask=f==k
                e=(z-point*w)[mask]
                r=rankdata(e)/(sum(mask)+1)
                raw=r-.5 if name=="Rank-W" else norm.ppf(r)
                # Within a fold, implementation averages *scores* over tied ranks.
                if len(np.unique(e))<len(e):
                    grid=norm.ppf(np.arange(1,len(e)+1)/(len(e)+1)) if name=="Rank-N" else np.arange(1,len(e)+1)/(len(e)+1)-.5
                    raw=np.array([grid[(rankdata(e,method="min")[j].astype(int)-1):rankdata(e,method="max")[j].astype(int)].mean() for j in range(len(e))])
                score[mask]=raw-raw.mean()
            expected=np.sqrt(np.mean(w*w*score*score)/len(w))/meta["slope"]
            self.assertAlmostEqual(ci["sandwich"]["se"],expected,places=11)

    def test_adaptive_reference_and_mixed(self):
        w,z,f=self.w,self.z,self.f
        old=common.rank_methods(w,z,f,self.pre,np.random.default_rng(18))["Rank-Ad"]
        point,ci,info=adaptive(w,z,f,self.pre,0,(1,2,3,4,5,6),np.random.default_rng(18),np.random.default_rng(19))
        self.assertAlmostEqual(point,old[0],places=11)
        self.assertAlmostEqual(ci["naive"]["se"],old[1],places=11)
        self.assertEqual(info["selected_L"],old[2]["selected_L"])
        self.assertAlmostEqual(info["max_t_critical"],old[2]["max_t_critical"],places=9)
        self.assertTrue(ci["sandwich"]["valid"] and ci["sandwich_max_t"]["valid"])
        p,_,diagnostics=regular_rank("Fixed mixed",w,z,f,self.pre,0,False)
        s=rank_values(w,z,f,self.pre,(1,2))
        expected=self.pre+(w@s.sum(1))/(len(w)*diagnostics["slope"])
        self.assertAlmostEqual(p,expected,places=12)

    def test_invalid_and_extreme_interval_handling(self):
        for se in (np.nan,np.inf,0,-1,None):
            i=interval(1.,se,0.)
            self.assertFalse(i["covered"])
            self.assertFalse(i["valid"])
            self.assertTrue(i["failure"])
        self.assertTrue(interval(1000.,1.,0.)["valid"])

class ReportingTests(unittest.TestCase):
    def test_failed_interval_remains_in_denominator(self):
        import pandas as pd
        from report import summarize
        common=dict(scenario="D1",n=500,method="Rank-W",inference="sandwich",
                    theta0=1.,pseudo_true=1.,seconds=.01,valid_point=True)
        cases=pd.DataFrame([
            common|dict(point=.99,covered_pseudo_true=True)|interval(.99,.1,1.),
            common|dict(point=1.01,covered_pseudo_true=False)|interval(1.01,np.nan,1.),
        ])
        result=summarize(cases).iloc[0]
        self.assertEqual(result.coverage,.5)
        self.assertEqual(result.valid_only_coverage,1.)
        self.assertEqual(result.valid_points,2)
        self.assertEqual(result.valid_intervals,1)
        self.assertEqual(result.failure_rate,.5)

if __name__=="__main__":
    unittest.main(verbosity=2)


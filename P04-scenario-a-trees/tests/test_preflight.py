"""Checks that do not fit any tree models or start any experiment."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import run
import numpy as np


class ScenarioATreeTests(unittest.TestCase):
    def test_frozen_sources_grid_and_seeds(self):
        self.assertEqual(len(run.science_signature()), 64)
        registry = run.seed_registry()
        self.assertEqual(set(registry), set(run.keys_for("main")))
        self.assertEqual(len(registry), 27000)
        self.assertEqual(sum(seed[0] == 2026111401 for seed in registry.values()), 10000)
        self.assertEqual(len(run.methods.METHODS), 9)
        modern_key = (500, "logistic", 0)
        with patch.object(run.methods, "generate", return_value="marker") as generate:
            self.assertEqual(run.generate("main", *modern_key)[0], "marker")
            generate.assert_called_once_with("main", *modern_key)
        historic_key = (1000, "gaussian", 0)
        n = historic_key[0]
        result = (np.zeros((n,10)), np.zeros(n), np.zeros((n,1)), np.zeros(n), np.zeros((n,1)), 1.)
        with patch.object(run.legacy, "generate", return_value=result) as generate:
            arrays, seed = run.generate("main", *historic_key)
            self.assertEqual(seed[0], 2026111401)
            generate.assert_called_once()
            self.assertEqual(np.bincount(arrays[-1]).tolist(), [200]*5)

    def test_models_and_no_outer_validation_leakage(self):
        rf = run.make_model("rf", 5).get_params()
        self.assertEqual((rf["n_estimators"], rf["min_samples_leaf"], rf["criterion"], rf["n_jobs"]),
                         (500, 5, "squared_error", 1))
        gb = run.make_model("gbdt", 6).get_params()
        self.assertEqual((gb["n_estimators"], gb["max_depth"], gb["learning_rate"], gb["loss"],
                          gb["validation_fraction"], gb["n_iter_no_change"], gb["tol"]),
                         (200, 3, .05, "squared_error", .10, 10, .0001))
        p = np.arange(60.).reshape(6,10)
        y = p[:, :2]
        mask = np.array([True, False, True, False, True, False])
        models = [Mock(n_estimators_=19), Mock(n_estimators_=23)]
        for model in models:
            model.fit.return_value = model
        with patch.object(run, "make_model", side_effect=models):
            fitted, iterations = run.fit_pair(p, y, mask, "gbdt", [5,6])
        self.assertEqual(fitted, models)
        self.assertEqual(iterations, [19,23])
        for j, model in enumerate(models):
            a, b = model.fit.call_args.args
            np.testing.assert_array_equal(a, p[mask])
            np.testing.assert_array_equal(b, y[mask,j])

    def test_cache_stage2_and_resume_on_temporary_fixture(self):
        n=30
        x=np.arange(n*10, dtype=float).reshape(n,10)/100
        folds=np.arange(n)%5
        d=np.sin(np.arange(n))
        y=np.cos(np.arange(n))+d
        data=(x,d,y,d,y,folds)
        def fake_fit(p, response, train, learner, seeds):
            return [], [19, 24]
        def fake_predict(models, p):
            return p[:,:2]*.1
        with patch.object(run, "generate", return_value=(data, ["fixture"])), \
             patch.object(run, "fit_pair", side_effect=fake_fit) as fit, \
             patch.object(run, "predict_pair", side_effect=fake_predict), \
             patch.object(run.legacy, "covariates", return_value=x), \
             patch.object(run.legacy, "direct_w", return_value=.5):
            m, payload=run.fit_nuisance(("gbdt","smoke",n,"gaussian",0))
        self.assertEqual(m["status"],"ok")
        self.assertEqual(fit.call_count,20)
        for fold in range(5):
            p, _, outer_train, *_ = fit.call_args_list[fold*4].args
            np.testing.assert_array_equal(p[outer_train],x[folds!=fold])
            for k in range(1,4):
                p, *_ = fit.call_args_list[fold*4+k].args
                np.testing.assert_array_equal(p,x[folds!=fold])
        expected=[dict(method=name) for name in run.methods.METHODS]
        with patch.object(run.methods,"estimate_all",return_value=expected) as estimate:
            second,_=run.stage2((m,payload))
        cache=estimate.call_args.args[0]
        np.testing.assert_array_equal(cache["folds"],folds)
        np.testing.assert_allclose(cache["w"],d-x[:,0]*.1,rtol=0,atol=0)
        self.assertEqual(second["rows"],expected)
        with TemporaryDirectory() as tmp, patch.object(run,"ROOT",Path(tmp)), \
             patch.object(run,"science_signature",return_value="fixture"):
            with run.write_db("smoke","gbdt",1) as db, db:
                db.execute("INSERT INTO records VALUES(?,?,?,?,?)",
                           (n,"gaussian",0,run.json_text(m),payload))
            with patch.object(run,"dispatch") as dispatch:
                run.run_stage("smoke","gbdt",1,[(n,"gaussian",0)],2,True)
                dispatch.assert_not_called()
            with patch.object(run,"dispatch") as dispatch:
                with self.assertRaisesRegex(ValueError,"lacks"):
                    run.run_stage("smoke","gbdt",2,[(n,"gaussian",1)],2,True)
                dispatch.assert_not_called()
            with run.read_db(run.db_path("smoke","gbdt",1),"smoke","gbdt",1) as db:
                self.assertEqual(json.loads(db.execute("SELECT metadata FROM records").fetchone()[0]),m)
            with self.assertRaisesRegex(ValueError,"provenance"):
                with run.read_db(run.db_path("smoke","gbdt",1),"smoke","rf",1):
                    pass


if __name__ == "__main__":
    unittest.main()

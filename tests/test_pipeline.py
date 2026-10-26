import unittest
import numpy as np
import pandas as pd
from freight_model import Features, RateModel
from train import load_data

class PipelineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev,cls.val,cls.template,cls.december=load_data()

    def test_temporal_windows_do_not_overlap(self):
        for start,end in [('2025-05-01','2025-07-01'),('2025-07-01','2025-09-01'),('2025-09-01','2025-11-01')]:
            tr=self.dev[self.dev.date<start]; te=self.dev[(self.dev.date>=start)&(self.dev.date<end)]
            self.assertLess(tr.date.max(),te.date.min())
            self.assertFalse(set(tr.load_id)&set(te.load_id))

    def test_feature_fit_does_not_learn_future_city(self):
        train=self.dev.head(300)
        transform=Features().fit(train)
        before=transform.coordinates_['pickup'].copy(deep=True)
        future=self.val.head(1).copy(); future['pickup']='Never seen'; future['pickup_lat']=np.nan; future['pickup_lon']=np.nan
        result=transform.transform(future)
        self.assertTrue(pd.isna(result.pickup_lat.iloc[0]))
        pd.testing.assert_frame_equal(before,transform.coordinates_['pickup'])

    def test_invalid_weight_is_missing_not_absolute_value(self):
        sample=self.dev.head(3).copy(); sample['weight']=[-32000,0,32000]
        result=Features().fit_transform(sample)
        self.assertTrue(result.weight.iloc[:2].isna().all())
        self.assertEqual(result.weight_missing.tolist(),[1,1,0])

    def test_missing_december_features_and_unknown_cities_predict(self):
        model=RateModel(core=True).fit(self.dev.head(1500),self.dev.head(1500).posted_rate)
        frame=self.december.head(2).copy(); frame.loc[frame.index[0],'pickup']='Never seen'
        p=model.predict(frame)
        self.assertTrue(np.isfinite(p).all() and (p>0).all())

    def test_export_ids_order_and_positive_values(self):
        result=pd.read_csv('validation_predictions.csv')
        self.assertEqual(list(result),['load_id','predicted_rate'])
        self.assertEqual(result.load_id.tolist(),self.template.load_id.tolist())
        self.assertEqual(len(result),12000)
        self.assertFalse(result.load_id.duplicated().any())
        self.assertTrue(np.isfinite(result.predicted_rate).all() and (result.predicted_rate>0).all())

    def test_december_preserves_fixed_inputs(self):
        self.assertEqual(len(self.december),31)
        self.assertTrue(self.december.pickup.eq('Lexington').all())
        self.assertTrue(self.december.delivery.eq('Fort Wayne').all())
        self.assertTrue(self.december.distance.eq(360).all())
        self.assertTrue(self.december.weight.eq(32000).all())
        self.assertTrue(self.december.equipment.eq('Dry Van').all())
        self.assertTrue(np.isfinite(self.december.predicted_rate).all())

if __name__=='__main__': unittest.main()

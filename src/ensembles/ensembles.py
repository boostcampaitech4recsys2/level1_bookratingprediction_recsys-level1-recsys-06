import numpy as np
import pandas as pd
from pyparsing import col
#from ..models._models import rmse

class Ensemble:
    def __init__(self, filenames:str, filepath:str):
        self.filenames = filenames
        self.output_list = []

        output_path = [filepath+filename+'.csv' for filename in filenames]
        self.output_frame = pd.read_csv(output_path[0]).drop('rating',axis=1)
        self.output_df = self.output_frame.copy()

        for path in output_path:
            self.output_list.append(pd.read_csv(path)['rating'].to_list())
        for filename,output in zip(filenames,self.output_list):
            self.output_df[filename] = output

    # Simple Weighted
    # 직접 weight를 지정하여, 앙상블합니다.
    def simple_weighted(self,weight:list):
        if not len(self.output_list)==len(weight):
            raise ValueError("model과 weight의 길이가 일치하지 않습니다.")
        if np.sum(weight)!=1:
            raise ValueError("weight의 합이 1이 되도록 입력해 주세요.")

        pred_arr = np.append([self.output_list[0]], [self.output_list[1]], axis=0)
        for i in range(2, len(self.output_list)):
            pred_arr = np.append(pred_arr, [self.output_list[i]], axis=0)
        result = np.dot(pred_arr.T, np.array(weight))
        return result.tolist()

    # Average Weighted
    # (1/n)의 가중치로 앙상블을 진행합니다.
    def average_weighted(self):
        weight = [1/len(self.output_list) for _ in range(len(self.output_list))]
        pred_weight_list = [pred*np.array(w) for pred, w in zip(self.output_list,weight)]
        result = np.sum(pred_weight_list, axis=0)
        return result.tolist()

    # Mixed 
    # Negative case 발생 시, 다음 순서에서 예측한 rating으로 넘어가서 앙상블합니다.
    def mixed(self):
        result = self.output_df[self.filenames[0]].copy()
        for idx in range(len(self.filenames)-1):
            pre_idx = self.filenames[idx]
            post_idx = self.filenames[idx+1]
            result[self.output_df[pre_idx]<1] = self.output_df.loc[self.output_df[pre_idx]<1,post_idx]
        return result.tolist()

    def cold_condition(self, n):
        if not len(self.output_list) == 2:
            raise ValueError("두개의 모델만을 넣어주세요.")
        cold_users = cold_start_set(n)
        result = []
        for i, uid, isbn, m1, m2 in self.output_df.itertuples():
            if uid in cold_users:
                result.append(m2)
            else:
                result.append(m1)
        return result

def cold_start_set(n: int=0) -> set():
    df = pd.read_csv('/opt/ml/workspace/level1_bookratingprediction_recsys-level1-recsys-06/data/user_review_count.csv')
    cold_users = set(df[df['count']<=n]['user_id'])
    return cold_users

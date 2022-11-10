from model import get_prediction
from data_load import context_data_loader,process_context_data
from model import load_model, get_prediction
import pandas as pd

def single_prediction(data,user_id,isbn)->pd.DataFrame:
    data = process_context_data(data,user_id,isbn)
    data = context_data_loader(data)
    model = load_model(data['field_dims'])
    pred = get_prediction(model,data['test_dataloader'])
    pred[0] = max(1,min(10,pred[0]))
    data['test']['pred'] = pred
    result_df = pd.merge(data['test'][['user_id','isbn','pred']],data['books'],on='isbn')
    return result_df[['isbn','book_title','book_author','category','pred']]


def top_k_prediction(data,user_id,k)->pd.DataFrame:
    data = process_context_data(data,user_id,batch=True)
    data = context_data_loader(data)
    model = load_model(data['field_dims'])
    pred = get_prediction(model,data['test_dataloader'])
    data['test']['pred'] = pred
    data['test'] = data['test'].sort_values(by='pred',ascending=False)
    data['test']['pred'] = data['test']['pred'].apply(lambda x: max(1,min(10,x)))
    result_df = pd.merge(data['test'][['user_id','isbn','pred']].iloc[:k],data['books'],on='isbn')
    return result_df[['isbn','book_title','book_author','category','pred']]
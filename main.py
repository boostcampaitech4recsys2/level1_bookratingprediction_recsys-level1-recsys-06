import time
import argparse
import json
import pandas as pd

from src import seed_everything

from src.ksy_data import context_data_load, context_data_split, context_data_loader
#from src.data import context_data_load, context_data_split, context_data_loader
from src.ksy_data import dl_data_load, dl_data_split, dl_data_loader
#from src.data import dl_data_load, dl_data_split, dl_data_loader
from src.data import image_data_load, image_data_split, image_data_loader
#from src.data import text_data_load, text_data_split, text_data_loader
from src.ksy_data import text_data_load, text_data_split, text_data_loader

from src import FactorizationMachineModel, FieldAwareFactorizationMachineModel
from src import DeepCrossNetworkModel, FFDCNModel
from src import CNN_FM
from src import DeepCoNN

import wandb

def main(args):
    seed_everything(args.SEED)
    
    ######################## DATA LOAD
    print(f'--------------- {args.MODEL} Load Data ---------------')
    if args.MODEL in ('FM', 'FFM'):
        data = context_data_load(args)
    elif args.MODEL in ('NCF', 'WDN', 'DCN'):
        data = dl_data_load(args)
    elif args.MODEL == 'FFDCN':
        data = context_data_load(args)
        # datadcn = dl_data_load(args)
    else:
        pass

    ######################## Train/Valid Split
    print(f'--------------- {args.MODEL} Train/Valid Split ---------------')
    if args.MODEL in ('FM', 'FFM'):
        data = data
        # if 문을 활용해서 kfold 쓸지 안쓸지 하면 될듯 (ex) 모델명을  kFM 따로 만들어주는 )
        data = context_data_split(args, data)
        data = context_data_loader(args, data)
        

    elif args.MODEL in ('NCF', 'WDN', 'DCN'):
        data = dl_data_split(args, data)
        data = dl_data_loader(args, data)
        

    elif args.MODEL == 'FFDCN':
        seed_everything(args.SEED)
        data = context_data_split(args,data)
        data = context_data_loader(args,data)
        # seed_everything(args.SEED)
        # datadcn = dl_data_split(args,datadcn)
        # datadcn = dl_data_loader(args,datadcn)

    else:
        pass

    ######################## Model
    print(f'--------------- INIT {args.MODEL} ---------------')
    if args.MODEL=='FM':
        model = FactorizationMachineModel(args, data)
    elif args.MODEL=='FFM':
        model = FieldAwareFactorizationMachineModel(args, data)
    elif args.MODEL=='DCN':
        model = DeepCrossNetworkModel(args, data)
    elif args.MODEL=='FFDCN':
        model = FFDCNModel(args, data)
    else:
        pass

    ######################## TRAIN
    print(f'--------------- {args.MODEL} TRAINING ---------------')
    # kfold 
    model.train() # model.kfold_train()

    ######################## INFERENCE
    print(f'--------------- {args.MODEL} PREDICT ---------------')
    if args.MODEL in ('FM', 'FFM', 'NCF', 'WDN', 'DCN'):
        predicts = model.predict(data['test_dataloader'])
    elif args.MODEL == 'FFDCN':
        predicts = model.predict(data['test_dataloader'])
    else:
        pass

    ####################### SAVE PREDICT
    print(f'--------------- SAVE {args.MODEL} PREDICT ---------------')
    submission = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')
    if args.MODEL in ('FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN', 'FFDCN'):
        submission['rating'] = predicts
    else:
        pass

    now = time.localtime()
    now_date = time.strftime('%Y%m%d', now)
    now_hour = time.strftime('%X', now)
    save_time = now_date + '_' + now_hour.replace(':', '')
    if args.MODEL == 'FFDCN':
        submission.to_csv('submit/{}_{}_{}{}{}{}{}{}{}{}.csv'.format(save_time, args.MODEL, args.USER_N, args.ISBN_N, args.AUTHOR_N, args.PUBLISH_N, args.CATEGORY_N, args.STATE_N, args.COUNTRY_N, args.CITY_N), index=False)
    else:
        submission.to_csv('submit/{}_{}.csv'.format(save_time, args.MODEL), index=False)



if __name__ == "__main__":

    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    ############### BASIC OPTION
    arg('--DATA_PATH', type=str, default='data/', help='Data path를 설정할 수 있습니다.')
    arg('--MODEL', type=str, choices=['FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN', 'FFDCN'],
                                help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--DATA_SHUFFLE', type=bool, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--TEST_SIZE', type=float, default=0.2, help='Train/Valid split 비율을 조정할 수 있습니다.')
    arg('--SEED', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    arg('-c','--config',default=None, type=str, help='지정된 경로의 JSON파일에서 설정값을 가져와 사용합니다.')
    
    ############### TRAINING OPTION
    arg('--BATCH_SIZE', type=int, default=1024, help='Batch size를 조정할 수 있습니다.')
    arg('--EPOCHS', type=int, default=10, help='Epoch 수를 조정할 수 있습니다.')
    arg('--LR', type=float, default=1e-3, help='Learning Rate를 조정할 수 있습니다.')
    arg('--WEIGHT_DECAY', type=float, default=1e-6, help='Adam optimizer에서 정규화에 사용하는 값을 조정할 수 있습니다.')

    ############### GPU
    arg('--DEVICE', type=str, default='cuda', choices=['cuda', 'cpu'], help='학습에 사용할 Device를 조정할 수 있습니다.')

    ############### FM
    arg('--FM_EMBED_DIM', type=int, default=16, help='FM에서 embedding시킬 차원을 조정할 수 있습니다.')

    ############### FFM
    arg('--FFM_EMBED_DIM', type=int, default=16, help='FFM에서 embedding시킬 차원을 조정할 수 있습니다.')

    ############### DCN
    arg('--DCN_EMBED_DIM', type=int, default=16, help='DCN에서 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--DCN_MLP_DIMS', type=list, default=(8, 8), help='DCN에서 MLP Network의 차원을 조정할 수 있습니다.')
    arg('--DCN_DROPOUT', type=float, default=0.2, help='DCN에서 Dropout rate를 조정할 수 있습니다.')
    arg('--DCN_NUM_LAYERS', type=int, default=3, help='DCN에서 Cross Network의 레이어 수를 조정할 수 있습니다.')

    ############### HOW COLUMNS OTHER N?
    arg('--USER_N', type=int, default=2, help='user_id others 기준 N 입력')
    arg('--ISBN_N', type=int, default=20, help='ISBN others 기준 N 입력')

    arg('--USER_N_D', type=int, default=2, help='user_id DCN 모델 others 기준 N 입력')
    arg('--USER_N_F', type=int, default=2, help='ISBN others DCN 모델 기준 N 입력')

    arg('--ISBN_N_D', type=int, default=20, help='user_id FFM 모델 others 기준 N 입력')
    arg('--ISBN_N_F', type=int, default=20, help='ISBN FFM 모델 others 기준 N 입력')

    arg('--AUTHOR_N', type=int, default=20, help='AUTHOR others 기준 N 입력')
    arg('--PUBLISH_N', type=int, default=20, help='PUBLISH others 기준 N 입력')
    arg('--CATEGORY_N', type=int, default=20, help='CATEGORY others 기준 N 입력')
    arg('--STATE_N', type=int, default=20, help='STATE others 기준 N 입력')
    arg('--COUNTRY_N', type=int, default=20, help='COUNTRY others 기준 N 입력')
    arg('--CITY_N', type=int, default=20, help='CITY others 기준 N 입력')

    args = parser.parse_args()

    args.MODEL = 'FFDCN'
    args.EPOCHS = 5
    args.DEVICE = 'cuda'
    # if args.config:
    #     # config 파일에서 인자 값들을 읽어온다.
    #     with open(args.config, 'rt') as f:
    #         t_args = argparse.Namespace()
    #         t_args.__dict__.update(json.load(f))
    #         args = parser.parse_args(namespace=t_args)
    main(args)

import time
import argparse
import json
import pandas as pd

from src import seed_everything

#from src.data import context_data_load, context_data_split, context_data_loader
from src.ksy_data import context_data_load, context_data_split, context_data_loader
from src.data import dl_data_load, dl_data_split, dl_data_loader
from src.data import image_data_load, image_data_split, image_data_loader
from src.data import text_data_load, text_data_split, text_data_loader
#from src.ksy_data import text_data_load, text_data_split, text_data_loader

from src import FactorizationMachineModel, FieldAwareFactorizationMachineModel
from src import NeuralCollaborativeFiltering, WideAndDeepModel, DeepCrossNetworkModel, FFDCNModel
from src import CNN_FM
from src import DeepCoNN
from src.data import textcon_data_load, textcon_data_loader, textcon_data_split
from src import FactorizationTextMachineModel
import wandb

def main(args):
    seed_everything(args.SEED)
    
    ######################## DATA LOAD
    print(f'--------------- {args.MODEL} Load Data ---------------')
    if args.MODEL in ('FM', 'FFM'):
        data = context_data_load(args)
    elif args.MODEL in ('NCF', 'WDN', 'DCN'):
        data = dl_data_load(args)
    elif args.MODEL == 'CNN_FM':
        data = image_data_load(args)
    elif args.MODEL == 'DeepCoNN':
        import nltk
        nltk.download('punkt')
        data = text_data_load(args)
    elif args.MODEL == 'FFDCN':
        dataffm = context_data_load(args)
        #datadcn = dl_data_load(args)
    else:
        data = textcon_data_load(args)
        pass

    ######################## Train/Valid Split
    print(f'--------------- {args.MODEL} Train/Valid Split ---------------')
    if args.MODEL in ('FM', 'FFM'):
        data = context_data_split(args, data)
        data = context_data_loader(args, data)

    elif args.MODEL in ('NCF', 'WDN', 'DCN'):
        data = dl_data_split(args, data)
        data = dl_data_loader(args, data)

    elif args.MODEL=='CNN_FM':
        data = image_data_split(args, data)
        data = image_data_loader(args, data)

    elif args.MODEL=='DeepCoNN':
        data = text_data_split(args, data)
        data = text_data_loader(args, data)
    elif args.MODEL == 'FFDCN':
        dataffm = context_data_split(args,dataffm)
        dataffm = context_data_loader(args,dataffm)
        # seed_everything(args.SEED)
        # datadcn = dl_data_split(args,datadcn)
        # datadcn = dl_data_loader(args,datadcn)
        
    else:
        data = textcon_data_split(args,data)
        data = textcon_data_loader(args,data)
        pass
    ######################## Model
    print(f'--------------- INIT {args.MODEL} ---------------')
    if args.MODEL=='FM':
        model = FactorizationMachineModel(args, data)
    elif args.MODEL=='FFM':
        model = FieldAwareFactorizationMachineModel(args, data)
    elif args.MODEL=='NCF':
        model = NeuralCollaborativeFiltering(args, data)
    elif args.MODEL=='WDN':
        model = WideAndDeepModel(args, data)
    elif args.MODEL=='DCN':
        model = DeepCrossNetworkModel(args, data)
    elif args.MODEL=='CNN_FM':
        model = CNN_FM(args, data)
    elif args.MODEL=='DeepCoNN':
        model = DeepCoNN(args, data)
    elif args.MODEL=='FFDCN':
        model = FFDCNModel(args,dataffm)
    else:
        model = FactorizationTextMachineModel(args, data)
        pass

    ######################## TRAIN
    print(f'--------------- {args.MODEL} TRAINING ---------------')
    model.train()

    ######################## INFERENCE
    print(f'--------------- {args.MODEL} PREDICT ---------------')
    if args.MODEL in ('FM', 'FFM', 'NCF', 'WDN', 'DCN'):
        predicts = model.predict(data['test_dataloader'])
    elif args.MODEL=='CNN_FM':
        predicts  = model.predict(data['test_dataloader'])
    elif args.MODEL=='DeepCoNN':
        predicts  = model.predict(data['test_dataloader'])
    elif args.MODEL == 'FFDCN':
        predicts = model.predict(dataffm['test_dataloader'])#, datadcn['test_dataloader'])
    else:
        predicts = model.predict(data['test_dataloader'])
        pass

    ######################## SAVE PREDICT
    print(f'--------------- SAVE {args.MODEL} PREDICT ---------------')
    submission = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')
    if args.MODEL in ('FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN','FFDCN'):
        submission['rating'] = predicts
    else:
        submission['rating'] = predicts
        pass

    now = time.localtime()
    now_date = time.strftime('%Y%m%d', now)
    now_hour = time.strftime('%X', now)
    save_time = now_date + '_' + now_hour.replace(':', '')
    submission.to_csv('submit/{}_{}.csv'.format(save_time, args.MODEL), index=False)



if __name__ == "__main__":

    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    ############### BASIC OPTION
    arg('--DATA_PATH', type=str, default='data/', help='Data path를 설정할 수 있습니다.')
    arg('--MODEL', type=str, choices=['FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN'],
                                help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--DATA_SHUFFLE', type=bool, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--TEST_SIZE', type=float, default=0.2, help='Train/Valid split 비율을 조정할 수 있습니다.')
    arg('--SEED', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    arg('-c','--config',default=None, type=str, help='지정된 경로의 JSON파일에서 설정값을 가져와 사용합니다.')
    
    ############### TRAINING OPTION
    arg('--BATCH_SIZE', type=int, default=1024, help='Batch size를 조정할 수 있습니다.')
    arg('--EPOCHS', type=int, default=7, help='Epoch 수를 조정할 수 있습니다.')
    arg('--LR', type=float, default=1e-3, help='Learning Rate를 조정할 수 있습니다.')
    arg('--WEIGHT_DECAY', type=float, default=1e-6, help='Adam optimizer에서 정규화에 사용하는 값을 조정할 수 있습니다.')

    ############### GPU
    arg('--DEVICE', type=str, default='cuda', choices=['cuda', 'cpu'], help='학습에 사용할 Device를 조정할 수 있습니다.')

    ############### FM
    arg('--FM_EMBED_DIM', type=int, default=16, help='FM에서 embedding시킬 차원을 조정할 수 있습니다.')

    ############### FFM
    arg('--FFM_EMBED_DIM', type=int, default=16, help='FFM에서 embedding시킬 차원을 조정할 수 있습니다.')

    ############### NCF
    arg('--NCF_EMBED_DIM', type=int, default=16, help='NCF에서 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--NCF_MLP_DIMS', type=list, default=(16, 16), help='NCF에서 MLP Network의 차원을 조정할 수 있습니다.')
    arg('--NCF_DROPOUT', type=float, default=0.2, help='NCF에서 Dropout rate를 조정할 수 있습니다.')

    ############### WDN
    arg('--WDN_EMBED_DIM', type=int, default=16, help='WDN에서 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--WDN_MLP_DIMS', type=list, default=(16, 16), help='WDN에서 MLP Network의 차원을 조정할 수 있습니다.')
    arg('--WDN_DROPOUT', type=float, default=0.2, help='WDN에서 Dropout rate를 조정할 수 있습니다.')

    ############### DCN
    arg('--DCN_EMBED_DIM', type=int, default=16, help='DCN에서 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--DCN_MLP_DIMS', type=list, default=(16, 16), help='DCN에서 MLP Network의 차원을 조정할 수 있습니다.')
    arg('--DCN_DROPOUT', type=float, default=0.2, help='DCN에서 Dropout rate를 조정할 수 있습니다.')
    arg('--DCN_NUM_LAYERS', type=int, default=3, help='DCN에서 Cross Network의 레이어 수를 조정할 수 있습니다.')

    ############### CNN_FM
    arg('--CNN_FM_EMBED_DIM', type=int, default=128, help='CNN_FM에서 user와 item에 대한 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--CNN_FM_LATENT_DIM', type=int, default=8, help='CNN_FM에서 user/item/image에 대한 latent 차원을 조정할 수 있습니다.')

    ############### DeepCoNN
    arg('--DEEPCONN_VECTOR_CREATE', type=bool, default=False, help='DEEP_CONN에서 text vector 생성 여부를 조정할 수 있으며 최초 학습에만 True로 설정하여야합니다.')
    arg('--DEEPCONN_EMBED_DIM', type=int, default=32, help='DEEP_CONN에서 user와 item에 대한 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--DEEPCONN_LATENT_DIM', type=int, default=10, help='DEEP_CONN에서 user/item/image에 대한 latent 차원을 조정할 수 있습니다.')
    arg('--DEEPCONN_CONV_1D_OUT_DIM', type=int, default=50, help='DEEP_CONN에서 1D conv의 출력 크기를 조정할 수 있습니다.')
    arg('--DEEPCONN_KERNEL_SIZE', type=int, default=3, help='DEEP_CONN에서 1D conv의 kernel 크기를 조정할 수 있습니다.')
    arg('--DEEPCONN_WORD_DIM', type=int, default=768, help='DEEP_CONN에서 1D conv의 입력 크기를 조정할 수 있습니다.')
    arg('--DEEPCONN_OUT_DIM', type=int, default=32, help='DEEP_CONN에서 1D conv의 출력 크기를 조정할 수 있습니다.')

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

    arg('--DCN_MLP_DIM_LAYERS', type=int, default=2, help='DCN 모델의 MLP 레이어 개수')
    arg('--DCN_MLP_DIM_NUM', type=int, default=2, help='DCN 모델의 MLP 레이어의 크기')

    args = parser.parse_args()

    if args.config:
        # config 파일에서 인자 값들을 읽어온다.
        with open(args.config, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)
    args.MODEL = 'FFDCN'
    #args.EPOCHS = 5
    #args.DCN_MLP_DIMS = [13,13,13]
    #args.WEIGHT_DECAY = 1.0216921879280201e-06
    #args.DCN_MLP_DIMS = [args.DCN_MLP_DIM_NUM] * args.DCN_MLP_DIM_LAYERS
    main(args)

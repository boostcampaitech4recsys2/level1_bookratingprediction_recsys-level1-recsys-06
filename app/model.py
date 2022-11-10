from _models import _FFDCNModel
import argparse
import json
import torch
import streamlit as st

@st.cache
def load_model(field_dims) -> _FFDCNModel:
    parser = argparse.ArgumentParser()
    args = argparse.Namespace()
    with open('config.json','rt') as f:
        args.__dict__.update(json.load(f))
    ff_embed_dim = args.FFM_EMBED_DIM
    dcn_embed_dim = args.DCN_EMBED_DIM
    epochs = args.EPOCHS
    learning_rate = args.LR
    weight_decay = args.WEIGHT_DECAY
    device = args.DEVICE
    mlp_dims = args.DCN_MLP_DIMS
    dropout = args.DCN_DROPOUT
    num_layers = args.DCN_NUM_LAYERS

    model = _FFDCNModel(field_dims, ff_embed_dim, dcn_embed_dim, num_layers=num_layers, mlp_dims=mlp_dims, dropout=dropout).to(device)
    model.load_state_dict(torch.load(args.MODEL_PATH, map_location=device))

    return model

def get_prediction(model, dataloader):
    model.eval()
    predicts = list()
    with torch.no_grad():
        for fields in dataloader:
            fields = fields[0].to('cuda')
            y = model(fields)
            predicts.extend(y.tolist())
    return predicts
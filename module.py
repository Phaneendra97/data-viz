import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torch
import transformers
import datasets
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import typing
# from shap import Explanation

TOKENIZER = AutoTokenizer.from_pretrained("bvanaken/CORe-clinical-diagnosis-prediction")
MODEL = AutoModelForSequenceClassification.from_pretrained("bvanaken/CORe-clinical-diagnosis-prediction")
PRED = transformers.pipeline("text-classification", model=MODEL, tokenizer=TOKENIZER, return_all_scores=True)
EXPLAINER = shap.Explainer(PRED)


def disease_shap_exp(prmot: str) -> typing.Tuple[shap.Explanation, typing.List[typing.Tuple[str, float]]]:
    """
    Give a prompt and it will return SHAP values.
    Returns ONLY the values that:
        * are numeric i.e. codes i.e. 401, 3440
        * and have prob > 0.3
    
    """
    # first making the model predictions
    tokenized_input = TOKENIZER(prmot, return_tensors="pt")
    output = MODEL(**tokenized_input)
    predictions = torch.sigmoid(output.logits)
    # predicted_labels: [('038', 0.9923), (..., ...)]
    ICD9_DIS = pd.read_csv("data/ICD/CMS32_DESC_LONG_SHORT_DX.csv", dtype={'DIAGNOSIS CODE': 'str'})
    ICD9_DIS['DIAGNOSIS CODE'] = ICD9_DIS['DIAGNOSIS CODE'].str.replace('.', '')

    # predicted_labels = [
    #     (MODEL.config.id2label[_id],
    #     ICD9_DIS.loc[ICD9_DIS['DIAGNOSIS CODE'].str.contains(f"^{MODEL.config.id2label[_id]}.*"), ' disease'].tolist()[0],
    #     float(predictions[0][_id]))
    #     for _id in np.argwhere(predictions.detach().numpy() > 0.3)[:, 1] if MODEL.config.id2label[_id].isnumeric()
    # ]
    pred_codes = []
    pred_probs = []
    long_desc = []
    short_desc = []
    for _id in np.argwhere(predictions.detach().numpy() > 0.3)[:, 1]:
        if MODEL.config.id2label[_id].isnumeric():
            pred_codes.append(MODEL.config.id2label[_id])
            pred_probs.append(float(predictions[0][_id]))
            long_desc.append(ICD9_DIS.loc[ICD9_DIS['DIAGNOSIS CODE'].str.contains(f"^{MODEL.config.id2label[_id]}.*"), 'LONG DESCRIPTION'].tolist()[0])
            short_desc.append(ICD9_DIS.loc[ICD9_DIS['DIAGNOSIS CODE'].str.contains(f"^{MODEL.config.id2label[_id]}.*"), 'SHORT DESCRIPTION'].tolist()[0])

    shap_value = EXPLAINER([prmot])

    return (shap_value[:, :, pred_codes], pred_codes, pred_probs, long_desc, short_desc)

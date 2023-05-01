import os
import torch
import torch.nn.functional as F
import pandas as pd
from collections import defaultdict


def model_predict(model,model_name,test_dataloader,device):


    res_dic = defaultdict(list)

    main_path = '/kaggle/input/baseline-model'
    model_path = os.path.join(main_path,model_name)

    # load model
    model.load_state_dict(torch.load(model_path,map_location=device))

    model.eval()

    with torch.no_grad():

        for idx,x in enumerate(test_dataloader):
            res_dic['row_id'].append(idx)

            logits = model(x)

            probs = F.softmax(logits,dim=-1)[0].numpy()
            
            for c,prob in enumerate(probs):
                res_dic[c].append(prob)

    res = pd.DataFrame(res_dic)

    return res




    
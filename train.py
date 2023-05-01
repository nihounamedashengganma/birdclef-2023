import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm


def model_eval(model,val_dataloader,device):

    model.eval()

    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for step,(x,y) in enumerate(val_dataloader):
            x,y = x.to(device),y.to(device)
            
            logits = model(x)

            batch_loss = criterion(logits,y)

            total_loss += batch_loss
        
        total_loss /= (step+1)
    
    return total_loss


def model_train(model,model_name,train_dataloader,val_dataloader,epochs,lr,device):

    model = model.to(device)

    # define optimizer
    optimizer = Adam(model.parameters(),lr = lr)
    # define loss func
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_epoch,min_loss = 0,1e10

    for epoch in range(1,epochs+1):
        # train mode
        model.train()

        train_loss = 0.0

        for step,(x,y) in tqdm(enumerate(train_dataloader)):
            x,y = x.to(device),y.to(device) 
      
            logits = model(x)

            batch_loss = criterion(logits,y)

            optimizer.zero_grad() # 清除梯度
            batch_loss.backward() # 计算梯度

            optimizer.step() # 反向传播

            train_loss += batch_loss
        
        train_loss /= (step+1)

        # eval
        eval_loss = model_eval(model,val_dataloader,device)

        print('at epoch {} the training loss is {} the eval loss is {}'.format(epoch,train_loss,eval_loss))

        if eval_loss < min_loss:
            min_loss = eval_loss
            best_epoch = epoch

            torch.save(model.state_dict(),f'{model_name}.pkl')
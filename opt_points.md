### 1.学习率设计

### 2.额外真实数据补充

### 3.多样的数据增强

### 4.调参 (img_size,smoothing_rate)

### 5.other tricks (e.g. flooding,grad_clip etc.)
```
outputs = model(inputs)
loss = criterion(outputs, labels)
flood = (loss-b).abs()+b # This is it!
optimizer.zero_grad()
flood.backward()
optimizer.step()
```

import pandas as pd

# 上边是给数据打上星期标识
# data = pd.read_csv('date_label.csv', engine='python')
# data=pd.DataFrame(data)
# data['x']=''
# x=[]
# for n in range(61):
#     # print(n)
#     x.append(1)
#     x.append(2)
#     x.append(3)
#     x.append(4)
#     x.append(5)
#     x.append(6)
#     x.append(7)
# data['x']=x
# data.to_csv('data.csv', encoding='utf-8', index=None)
# print(data)

data = pd.read_csv('data2.csv', engine='python')
tp=data['total_purchase_amt'].copy()
tr=data['total_redeem_amt'].copy()
x=data['x'].copy()
p=data['p'].copy()
r=data['r'].copy()

for i in range(426):
    if p[i] > 0.352852616:
        if x.index[i] not in ['1','6','7']:
            tp[i]=(tp[i-1]+tp[i+1]+tp[i])/3

for i in range(426):
    if r[i] > 0.446706706:
        if x[i] not in ['1','6','7']:
            tr[i]=(tr[i-1]+tr[i+1]+tr[i])/3

data['total_purchase_amt']=tp
data['total_redeem_amt']=tr
data.to_csv('data3.csv', encoding='utf-8', index=None)
print(data)
print(tp)
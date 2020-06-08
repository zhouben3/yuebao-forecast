import pandas as pd
import numpy as np
from dtw import dtw
data = pd.read_csv('databo.csv', engine='python')
tp=data['total_purchase_amt'].copy()
tr=data['total_redeem_amt'].copy()
date=data['report_date']
p=data['p'].copy()
r=data['r'].copy()

# print(p[426])
min=1000
mindata=0

i=4
a=1
while a<=427-i:
    tmp = []
    for k in range(i):
        tmp.append(p[a+k])
    x=  np.array(tmp).reshape(-1, 1)
    # print(x)
    y=np.array([-0.153593571,-0.423835128,0.483094534,0.280651105]).reshape(-1, 1)
    manhattan_distance = lambda x, y: np.abs(x - y)
    d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=manhattan_distance)
    if d< min:
        min=d
        mindata=date[a]
    a=a+1
print(min,mindata)





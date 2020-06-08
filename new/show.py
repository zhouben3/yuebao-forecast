import pandas
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
data = pandas.read_csv('data3.csv', engine='python')

plt.rcParams['figure.figsize'] = (25, 4.0)  # set figure size
data.rename(columns={'total_purchase_amt':'申购', 'total_redeem_amt':'赎回'}, inplace = True)
data[['申购', '赎回']].plot()
plt.grid(True, linestyle="-", color="green", linewidth="0.5")
plt.legend()
plt.xlabel("天数")
plt.ylabel("资金量")
plt.title('申购赎回趋势图')

plt.gca().spines["top"].set_alpha(0.0)
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.0)
plt.gca().spines["left"].set_alpha(0.3)

plt.show()
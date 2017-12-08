import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
plt.style.use('fivethirtyeight')

# Product Forge Future Health demand estimation - Mike Walmsley and Team PIM

days = 120
t = np.arange(0,days) # t in day units

# generate synthetic data

# define feature sets
# each feature represents an underlying known demand driver e.g. operations, admits, season
# mathematically, this is the known independent (but potentially correlated) variables in the regression

feat1_w = 0.02
feat1_scale = 0.3
feat1_phi = -90
feat1 = feat1_scale *( np.sin(feat1_w * t + feat1_phi)*np.sin(feat1_w * t + feat1_phi) )

feat2_base = 0.5
feat2_factor = 0.2
feat2_phi = -90
feat2_w = 0.04
feat2_scale = 0.4
feat2 = feat2_scale * ( feat2_base + feat2_factor*np.cos(feat2_w * t + feat2_phi) )

feat3 = np.ones_like(feat1) * 0.3

# y is true demand, from combining underlying known demand trends
y = feat1 + feat2 + feat3
y = y + y.min() # always positive
Y = y + np.random.normal(0,0.4,np.shape(y)) # Y (capital) is measured demand, subject to high noise
# noise represents random variation by day
# mathematically, Y is the dependent variable in the regression. Past Y is known, future Y is not.

# visualise underlying demand trends and true demand
plt.plot(t, feat1)
plt.plot(t, feat2)
plt.plot(t, feat3)
# plt.show()
plt.plot(t, y)
# plt.ylim([0,1])
plt.legend(['feat1','feat2','feat3','demand'],loc=0)
plt.title('Synthetic Underlying Demand Trends')
plt.ylabel('Product Units')
plt.xlabel('Day')
plt.show()

# Model will estimate the true demand y from the known demand drivers feat1, feat2, feat3, by day.
X = np.stack((feat1, feat2, feat3), axis=1)


y_sq = np.ones_like(y) * np.sum(Y) / days # y_sq is routine delivery model i.e. flat estimated demand - status quo

reg = linear_model.BayesianRidge()
reg.fit(X, Y)
y_est = reg.predict(X) # y_est is our model's estimation of true demand by day, inferred from underlying trends

# visualise status quo estimation of true demand vs. our estimation of true demand
# plt.subplot(211)
plt.plot(y_sq)
plt.plot(t, y_est)
plt.plot(t, Y)
plt.plot(t,y)
plt.title('Measured and Estimated Demand')
plt.ylabel('Products Required')
plt.xlabel('Day')
plt.legend(['NHS estimate', 'PIM estimate', 'Measured demand', 'True demand'],loc=0)

plt.show()

# if NHS restocked every 30 days according to each demand level, what would the stock levels look like?
diff_sq = y_sq - y
month_len = 30
months = int(days/month_len)

stock_sq_actual = np.zeros(month_len * months)
stock_est_actual = np.zeros(month_len * months)
for month in range(months):
    stock_sq = np.sum(y_sq[month*month_len:(month+1)*month_len])
    stock_est = np.sum(y_est[month*month_len:(month+1)*month_len])
    for day in range(month_len):
        total_day = month*month_len + day
        running_demand = np.sum( Y[month*month_len:month*month_len+day] )
        stock_sq_actual[total_day] = stock_sq - running_demand
        stock_est_actual[total_day] = stock_est - running_demand
# visualise
# plt.subplot(211)
plt.plot(stock_sq_actual)
plt.plot(stock_est_actual)
plt.plot(np.zeros_like(y_sq))
plt.title('Stock Level Comparison')
plt.legend(['Stock with NHS estimate', 'Stock with PIM estimate'])
plt.ylabel('Products in Stock')
plt.xlabel('Day')

plt.show()

final_stock_sq = -1 * np.ones(months)
final_stock_est =  -1 * np.ones(months)
months = np.arange(4)

for month in months:
    final_stock_sq[month] = stock_sq_actual[(month+1)*month_len-1]
    final_stock_est[month] = stock_est_actual[(month+1)*month_len-1]

width = 0.35       # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(months, final_stock_sq, width,color='b')
rects2 = ax.bar(months+width, final_stock_est, width,color='r')
ax.set_xticks(months + width)
ax.set_xticklabels(('Jan', 'Feb', 'Mar', 'Apr'))
plt.title('End of Month Stock')
plt.ylabel('Products in Stock')
plt.legend(['Stock with NHS estimate', 'Stock with PIM estiamte'],loc=0)
plt.show()
# print(final_stock_sq)
# print(final_stock_est)

np.savetxt('Y_true.txt', y)
np.savetxt('Y.txt', Y)
np.savetxt('Y_PIM.txt', y_est)
np.savetxt('Y_NHS.txt', y_sq)
np.savetxt('Stock_NHS.txt', stock_sq_actual)
np.savetxt('Stock_PIM.txt', stock_est_actual)





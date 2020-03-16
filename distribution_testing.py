#%%
# import necessary packages
import matplotlib.pyplot as plt
import powerlaw
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from preprocessing_data import df_time_series
register_matplotlib_converters()

df_data = df_time_series.copy()


#%%
# Power Law Distribution Testing
# df_returns = df_data['returns'].where(df_data['returns'] > 0).dropna(axis='rows')
arr_returns = df_data.returns.to_numpy()
fit = powerlaw.Fit(arr_returns)
alpha = fit.alpha
sigma = fit.sigma

fig2 = fit.plot_pdf(color='b', linewidth=2)
fit.power_law.plot_pdf(color='r', linestyle='--', ax=fig2)
plt.title('PDF - Power Law Fit')
plt.show()

R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)

print('Alpha: ' + str(alpha) + " Sigma: " + str(sigma) + " R: " + str(R) + " P-value: " + str(p))

fit.distribution_compare('power_law', 'exponential')
fig4 = fit.plot_ccdf(linewidth=3)
fit.power_law.plot_ccdf(ax=fig4, color='r', linestyle='--')
fit.exponential.plot_ccdf(ax=fig4, color='g', linestyle=':')
plt.title('Power Law vs Exponential Distribution')
plt.show()

#%%
# Show histogram for all columns of df
print("Distribution")
hist = df_data.hist()
plt.show()

hist_01 = df_data.hist(bins=50)
plt.show()

# Returns
nb_bins = 100
n, bins, patches = plt.hist(df_data['returns'], bins=nb_bins, density=True)
plt.title('Histogram - Absolute Returns - ' + str(nb_bins) + ' bins')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.show()
#%%
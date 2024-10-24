# %%
import pandas as pd
import matplotlib.pyplot as plt


accor_data = pd.read_csv('accor.csv')
airliquid_data = pd.read_csv('airliquid.csv')


accor_data['date'] = pd.to_datetime(accor_data['date'], errors='coerce')
airliquid_data['date'] = pd.to_datetime(airliquid_data['date'], errors='coerce')


accor_prices = accor_data[['date', 'clot']].dropna()
airliquid_prices = airliquid_data[['date', 'clot']].dropna()


accor_prices.rename(columns={'clot': 'Accor'}, inplace=True)
airliquid_prices.rename(columns={'clot': 'AirLiquide'}, inplace=True)


merged_data = pd.merge(accor_prices, airliquid_prices, on='date', how='inner')


merged_data['Accor'] = pd.to_numeric(merged_data['Accor'], errors='coerce')
merged_data['AirLiquide'] = pd.to_numeric(merged_data['AirLiquide'], errors='coerce')


merged_data['Accor_Returns'] = merged_data['Accor'].pct_change()
merged_data['AirLiquide_Returns'] = merged_data['AirLiquide'].pct_change()


returns_data = merged_data.dropna(subset=['Accor_Returns', 'AirLiquide_Returns'])


plt.figure(figsize=(10,6))
plt.plot(returns_data['date'], returns_data['Accor_Returns'], label='Accor Returns', color='blue', alpha=0.7)
plt.plot(returns_data['date'], returns_data['AirLiquide_Returns'], label='AirLiquide Returns', color='green', alpha=0.7)


plt.title('Daily Returns of Accor and Air Liquide Stocks')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()


plt.grid(True)
plt.tight_layout()
plt.show()

# %%

# %% 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

df = pd.read_csv('./cellnet-vs-countess-all.csv')
a = '#Cells (Countess)'; b = '#Cells (CellNet)'
df = df.rename(columns={'Countess': a, 'CellNet': b})

pearsonr(df[a], df[b])

# add a label for the Pearson correlation coefficient
# add a linear regression line with confidence interval
r, p = pearsonr(df[a], df[b])
sns.regplot(x=a, y=b, data=df, scatter=True, label=f'Pearson r = {r:.2f}, p = {p:.2e}')

sns.set_style('whitegrid')
#plt.title('Comparison of Cell Counts between CellNet and Countess')
plt.legend()

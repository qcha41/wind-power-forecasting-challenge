import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('runs.csv').set_index('Run ID')

run_list = ['0a43f348081f45af851997ebafe82619',
            'acbfafbf830a4742853f7bedf868ed93',
            '751ede7795d048aab8026cd181329f66',
            '3787663df804496189e24d5e3b78587f',
            '35837339511c40b38a62919817c77d16',
            '9f98f2b57ce843848dff6839b2a22cc0',
            '815b8a9e1ff84dd09d55b681fe551099',
            'cd560c945c144fbe9550f53903550f31']

df = df.loc[run_list].sort_values(by='units')

fig, ax = plt.subplots(figsize=(9,5))
ax.semilogx(df.units, df.loss)
ax.fill_between(df.units, df.loss-df.loss_std, df.loss+df.loss_std, alpha=0.2)
ax.semilogx(df.units, df.val_loss)
print(df.val_loss)
ax.fill_between(df.units, df.val_loss-df.val_loss_std, df.val_loss+df.val_loss_std, alpha=0.2)
# print(df.columns)
# print(df.loc[run_list,:])

plt.show()
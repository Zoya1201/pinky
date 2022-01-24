#!/usr/bin/env python
# coding: utf-8

# # Content with notebooks
# 
# You can also create content with Jupyter Notebooks. This means that you can include
# code blocks and their outputs in your book.
# 
# ## Markdown + notebooks
# 
# As it is markdown, you can embed images, HTML, etc into your posts!
# 
# ![](https://myst-parser.readthedocs.io/en/latest/_static/logo-wide.svg)
# 
# You can also $add_{math}$ and
# 
# $$
# math^{blocks}
# $$
# 
# or
# 
# $$
# \begin{aligned}
# \mbox{mean} la_{tex} \\ \\
# math blocks
# \end{aligned}
# $$
# 
# But make sure you \$Escape \$your \$dollar signs \$you want to keep!
# 
# ## MyST markdown
# 
# MyST markdown works in Jupyter Notebooks as well. For more information about MyST markdown, check
# out [the MyST guide in Jupyter Book](https://jupyterbook.org/content/myst.html),
# or see [the MyST markdown documentation](https://myst-parser.readthedocs.io/en/latest/).
# 
# ## Code blocks and outputs
# 
# Jupyter Book will also embed your code blocks and output in your book.
# For example, here's some sample Matplotlib code:

# In[1]:


from matplotlib import rcParams, cycler
import matplotlib.pyplot as plt
import numpy as np
plt.ion()


# In[2]:


# Fixing random state for reproducibility
np.random.seed(19680801)

N = 10
data = [np.logspace(0, 1, 100) + np.random.randn(100) + ii for ii in range(N)]
data = np.array(data).T
cmap = plt.cm.coolwarm
rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                Line2D([0], [0], color=cmap(.5), lw=4),
                Line2D([0], [0], color=cmap(1.), lw=4)]

fig, ax = plt.subplots(figsize=(10, 5))
lines = ax.plot(data)
ax.legend(custom_lines, ['Cold', 'Medium', 'Hot']);


# There is a lot more that you can do with outputs (such as including interactive outputs)
# with your book. For more information about this, see [the Jupyter Book documentation](https://jupyterbook.org)

# In[3]:


import pandas as pd

df = pd.DataFrame(
    {'condition_1': cond_1, 'condition_2': cond_2}, 
    index=np.arange(100)
)

df[:10]


# In[11]:


import matplotlib.pyplot as plt

plt.scatter(cond_1, cond_2, alpha=.6)
plt.xlabel('condition 1')
plt.ylabel('condition 2')
plt.title('Scatterplot')
plt.show()


# In[2]:


from statsmodels.stats.weightstats import ttest_ind

t_value, p_value, df = ttest_ind(cond_1, cond_2)

print(f'Obtained t-value: {t_value}')
print(f'Obtained p-value: {p_value}')


# In[9]:


import numpy as np

cond_1 = np.random.rand(100)
print(f'Condition 1 = {cond_1[:10]}')

cond_2 = cond_1 + (np.random.rand(100))
print(f'Condition 2 = {cond_2[:10]}')


# In[3]:


from IPython.display import display
import ipywidgets as widgets

slider = widgets.IntSlider()
display(slider)


# In[4]:


button = widgets.Button(description="Click Me!")
output = widgets.Output(layout={'border': '3px solid black'})

display(button, output)

def on_button_clicked(b):
    with output:
        print("Button clicked.")
        output
        
button.on_click(on_button_clicked)


# In[14]:


from IPython.display import Image
Image("flower.jpg")


# In[6]:


widgets.DatePicker(
    description='Pick a Date',
    disabled=False
)


# In[7]:


from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual

def f(x):
    return x

interact(f, x='Type something here');


# In[8]:


widgets.IntSlider(
    value=7,
    min=0,
    max=10,
    step=1,
    description='Test:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)


# In[9]:


widgets.FloatLogSlider(
    value=10,
    base=10,
    min=-10, # max exponent of base
    max=10, # min exponent of base
    step=0.2, # exponent step
    description='Log Slider'
)


# In[11]:


import plotly.io as pio
import plotly.express as px
import plotly.offline as py

df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species", size="sepal_length")
fig


# In[12]:


import plotly.express as px
df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
fig.show()


# In[13]:


import plotly.express as px
df = px.data.tips()
fig = px.bar(df, x="sex", y="total_bill", color="smoker", barmode="group")
fig.show()


# In[16]:


from IPython.display import YouTubeVideo

SalidaVideo = widgets.Output()

with SalidaVideo:
    display(YouTubeVideo('FAb39v8bkAg', width=500, height=500))
    
Salidas = widgets.HBox([SalidaVideo])
display(Salidas)


# In[ ]:





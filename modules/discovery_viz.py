import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/gdrive')

df = pd.read_csv('/content/gdrive/MyDrive/ECE 143 Group 18 Project/NASA_planetary_data.csv', skiprows=168)
df.head()

# Count the occurrences of each telescope and instrument
telescope_counts = df['disc_telescope'].value_counts()
instrument_counts = df['disc_instrument'].value_counts()

# Select the top 5 telescopes, and group the rest into 'Others'
top5_telescopes = telescope_counts.head(5)
telescope_others = pd.Series(telescope_counts[5:].sum(), index=['Others'])
telescope_counts = top5_telescopes.append(telescope_others)

# Select the top 5 instruments, and group the rest into 'Others'
top5_instruments = instrument_counts.head(5)
instrument_others = pd.Series(instrument_counts[5:].sum(), index=['Others'])
instrument_counts = top5_instruments.append(instrument_others)

plt.style.use('dark_background')

# Create a pie chart for telescopes
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)  # Subplot for telescopes
plt.pie(telescope_counts, labels=telescope_counts.index, autopct='%1.1f%%', startangle=90, colors= ['#636EFA',  '#EF553B',  '#00CC96',  '#AB63FA',  '#FFA15A',  '#19D3F3',  '#FF6692',  '#B6E880',  '#FF97FF', '#FECB52'])
plt.title('Distribution of Telescopes')

# Create a pie chart for instruments
plt.subplot(1, 2, 2)  # Subplot for instruments
plt.pie(instrument_counts, labels=instrument_counts.index, autopct='%1.1f%%', startangle=90, colors =['#636EFA',  '#EF553B',  '#00CC96',  '#AB63FA',  '#FFA15A',  '#19D3F3',  '#FF6692',  '#B6E880',  '#FF97FF','#FECB52'])
plt.title('Distribution of Instruments')

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

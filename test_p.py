
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv('cars.csv')

print(df.head(30))


df['exterior_color'] = df['exterior_color'].map(lambda ec: ec.replace(' ', '').lower() if not pd.isna(ec) else ec)

profile = ProfileReport(df, title="Cars Profiling Report")
profile.to_file("cars_report.html")
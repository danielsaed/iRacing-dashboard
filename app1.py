from iracingdataapi.client import irDataClient
import pandas as pd

# Reemplaza con tu email y contraseña de iRacing
username = 'danielsaed99@hotmail.com'
password = 'F3ChecoPerez11'
idc = irDataClient(username=username, password=password)

# ------------------------------------
a = idc.driver_list(2)

df = pd.DataFrame(a)

print(df.head(10))
from iracingdataapi.client import irDataClient
import pandas as pd

# Reemplaza con tu email y contraseña de iRacing
username = 'danielsaed99@hotmail.com'
password = ''
idc = irDataClient(username=username, password=password)

# ------------------------------------

import pandas as pd

files_index = {1:'OVAL.csv',5:'ROAD.csv', 6:'FORMULA.csv',4:'DROAD.csv', 3:'DOVAL.csv'}
for index in files_index:
    df = pd.DataFrame(idc.driver_list(index))
    df.to_csv(r"data/"+files_index[index])
    print(f"Processed " + files_index[index])


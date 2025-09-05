from iracingdataapi.client import irDataClient
import pandas as pd
import os

# Obtener credenciales desde variables de entorno
username = 'danielsaed99@hotmail.com'
password = os.getenv('PASS')  # Obtendrá el valor desde GitHub Secrets

if not password:
    raise ValueError("Password not found in environment variables!")

idc = irDataClient(username=username, password=password)

# ------------------------------------

import pandas as pd

# Crear directorio data si no existe
os.makedirs('data', exist_ok=True)

files_index = {1:'OVAL.csv', 5:'ROAD.csv', 6:'FORMULA.csv', 4:'DROAD.csv', 3:'DOVAL.csv'}

print("🏁 Starting iRacing data update...")
print(f"Username: {username}")
print("=" * 50)

for index in files_index:
    try:
        print(f"📥 Downloading {files_index[index]}...")
        df = pd.DataFrame(idc.driver_list(index))
        
        # Guardar en el directorio data
        filepath = os.path.join("data", files_index[index])
        df.to_csv(filepath, index=False)
        
        print(f"✅ Processed {files_index[index]} - {len(df)} records")
        
    except Exception as e:
        print(f"❌ Error processing {files_index[index]}: {str(e)}")
        continue

print("=" * 50)
print("🎉 Data update completed!")

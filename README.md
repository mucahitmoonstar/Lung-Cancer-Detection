# Yapay-Zeka-Projeleri
PYTORCH-----KERAS----LİNEER REGRESYON----KNN




# @title DATASET  HAKKINDA BİLGİLER
#DATASETİN ANLAMLANDIRILMASI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("/content/drive/MyDrive/survey lung cancer.csv")
print(f"DATASETİN BOYUTU : {data.shape}")
print("-----------------------------------------------------------------------")
print("--------------------DATASETİN SUTUNLARININ İSİMLERİ VE TÜRLERİ---------------")
print(f"-------------------DATASETİN SUTUNLARININ İSİMLERİ VE TÜRLERİ -----------{data.info()}")
print(""""    GENDER (Cinsiyet): Kişinin erkek veya kadın olduğunu belirtir.
    AGE (Yaş): Kişinin yaşını belirtir. Akciğer kanseri yaşla birlikte görülme sıklığı artan bir kanser türüdür.
    SMOKING (Sigara Kullanımı): Kişinin sigara içip içmediğini veya ne sıklıkla içtiğini belirtebilir. Sigara içmek akciğer kanserinin en önemli risk faktörlerinden biridir.

Diğer Değişkenlerin Muhtemel Anlamları:

    YELLOW_FINGERS (Sarı Tırnaklar): Sigara kullanımına bağlı olarak tırnakların sararması durumu.
    ANXIETY (Kaygı): Kişinin kaygı düzeyini belirtebilir. Stres ve kaygı bazı kanserlerle ilişkilendirilmiştir ancak akciğer kanseri için net bir bağlantı henüz kanıtlanmamıştır.
    PEER_PRESSURE (Akran Baskısı): Kişinin sigara içmeye veya sağlıksız alışkanlıklara yönelten baskı altında olup olmadığını belirtebilir.
    CHRONIC DISEASE (Kronik Hastalık): Kişinin herhangi bir kronik hastalığı olup olmadığını belirtir. Bazı kronik hastalıklar akciğer kanserine yakalanma riskini arttırabilir.

Sağlık Belirtileri Olarak Değişkenler:

    FATIGUE (Yorgunluk): Kişinin kendini sürekli yorgun hissetmesi. Akciğer kanseri ve birçok hastalıkta görülebilen bir belirtidir.
    ALLERGY (Alerji): Kişinin herhangi bir alerjisi olup olmadığını belirtebilir. Alerji ile akciğer kanseri arasında net bir ilişki bulunamamıştır.
    WHEEZING (Hırıltı): Solunum sırasında hırıltı sesi olması. Akciğer kanseri ve diğer solunum yolu hastalıklarında görülebilir.
    ALCOHOL CONSUMPTION (Alkol Tüketimi): Kişinin alkol tüketim alışkanlığını belirtebilir. Aşırı alkol tüketimi akciğer kanseri riskini arttırabilir.
    COUGHING (Öksürük): Kişinin öksürük şikayeti olup olmadığını belirtir. Akciğer kanseri ve birçok solunum yolu hastalığında görülebilen bir belirtidir.
    SHORTNESS OF BREATH (Nefes Darlığı): Kişinin nefes darlığı çekip çekmediğini belirtir. Akciğer kanseri ve diğer solunum yolu hastalıklarında görülebilir.
    SWALLOWING DIFFICULTY (Yutma Güçlüğü): Kişinin yutma güçlüğü çekip çekmediğini belirtir. Akciğer kanseri nadir olarak yutma güçlüğüne sebep olabilir ancak daha çok yemek borusu kanseri gibi diğer kanserlerde görülür.
    CHEST PAIN (Göğüs Ağrısı): Kişinin göğüs ağrısı şikayeti olup olmadığını belirtir. Akciğer kanseri ve diğer akciğer hastalıklarında görülebilir.

LUNG_CANCER (Akciğer Kanseri): Bu değişken ise kişide akciğer kanseri olup olmadığını belirten hedef değişkendir. Diğer değişkenlerden elde edilen bilgiler kullanılarak akciğer kanseri olup olmadığı tahmin edilmeye çalışılıyor olabilir.""")

print("-----------------------------------------------------------------------")
print("--------------ÖNCELİKLE DATASETİMİZDEKİ VERİLERİ SIFIRLIYORUZ-------------")
# Bu komut, veri setindeki her sütunda bulunan eksik değerlerin sayısını hesaplar.
data.isna().sum().to_frame().T.style.set_properties(**{"background-color": "#0750b7","color":"white","border": "3.5px  solid black"})
#adlı orijinal veri setinin bir kopyasını oluşturur ve bunu data_temp adlı
#yeni bir değişkene atar. Bu, orijinal veri setinin değiştirilmeden kalmasını sağlar.
data_temp = data.copy()
#Her bir sütunda 1 olan değerleri "Yes" ve 0 olan değerleri "No" ile değiştirir.
for column in data_temp.columns:
    data_temp[column] = data_temp[column].replace({1: "Yes" , 0 : "No"})
#data_temp adlı veri setinin ilk 5 satırını döndürür.
data_temp.head().style.set_properties(**{"background-color": "#2a9d8f","color":"white","border": "1.5px  solid black"})




print("-----------------DATASETİMİZDEKİ AĞIRLIKTA BULUNAN YAŞLARI EKRANA  YAZDIRALIM-----------------")

print(" 1---------------------------------------------------ÖNCELİKLE  DOĞRUMUZDA GENEL AKCİĞER KANSERİ OLAN YAŞLARI  BELİRTTİK--------------------------------------------------")

#veri setindeki "AGE" (Yaş) sütunu ile "LUNG_CANCER"
#(Akciğer Kanseri) sütunu arasındaki ilişkiyi görselleştirmek için bir histogram oluşturur.
plt.subplots(figsize=(20, 8))
p = sns.histplot(data, x="AGE", hue="LUNG_CANCER", multiple="stack", kde=True, shrink=.99, bins=20, alpha=1, fill=True)
p.legend_.set_title("Lung Cancer")

p.axes.set_title("\nAKCİĞER KANSERİ VE YAŞ İLİŞKİSİ \n", fontsize=20)
plt.ylabel("Count")
plt.xlabel("Age")

sns.despine(left=True, bottom=True)
plt.show()

print("*********************************************************************   1=ERKEK  0=KIZ  BİREYLERİ  GÖSTERMEKTEDİR    ***************************************************************************")




import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from termcolor import colored

# Dosya yolu ve adını belirtin
file_path = "/content/drive/MyDrive/survey lung cancer.csv"

# Veri setini yükleyin
data = pd.read_csv(file_path)

# Sütun adlarını kontrol edelim
print("Veri setindeki sütun adları:")
print(data.columns)

# Histogramlar için bir alt grafik oluşturun
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 20))

print(colored("------------------------------------------------------------VERİMİZDEKİ SÜTUNLARDA BULUNAN VERİLERİ GÖRSELLEŞTiRiYORUZ-----------------------------------------------------------", "red"))

print("/////////////////GENDER SÜTÜNUNDA BULUNAN ERKEK VE KADIN SAYISINI İLK GRAFİKTE EKRANA YAZDIRALIM /////////////////////////////////////////////////")
print("VERİLERİM GENEL OLARAK 1 VE SIFIRDAN OLUŞMAKTA 1 = MEVCUT YANİ HASTA 0 = MEVCUT DEĞİL YANİ SAĞLIKLI ")
# veri setindeki farklı özelliklerin dağılımını görselleştirmek için histograflar oluşturur. Veri setindeki
#her sütun için ayrı ayrı histogramlar çizilir, KDE eğrileri eklenir ve sütun adlarıyla etiketlenir.
# Bu görselleştirme, verilerin dağılımını ve olası ilişkileri anlamayı kolaylaştırır.
# Sütun adlarını ve sıralarını alın
columns = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE','FATIGUE ','ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN', 'LUNG_CANCER']

# Her bir sütun için histogramları çizin
for i, column in enumerate(columns):
    if column in data.columns:
        row = i // 4
        col = i % 4
        sns.histplot(data=data, x=column, ax=axes[row, col], kde=True)
        axes[row, col].set_title(column)
    else:
        print(f"{column} sütunu veri setinde bulunamadı.")

plt.tight_layout()
plt.show()
     #Kernel Density Estimate (KDE) eğrisi, sürekli bir olasılık dağılımının tahminini sağlayan bir istatistiksel
     #tekniktir. KDE, veri setindeki noktaların yoğunluğunu yumuşak bir şekilde temsil ederek, verilerin olasılık
    #yoğunluk fonksiyonunu tahmin eder.
print("-----------------------------------------------1 = BU KONUDA POZİTİF OLAN BİREYLERİ GÖSTERİYOR ÖRNEĞİN ='SİGARA İÇMEYEN BİREYLER 0 OLARAK NUMARALANDIRILMIŞTIR'-----------------------------------------")




import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

# Veri setini yükleme
file_path = "/content/drive/MyDrive/survey lung cancer.csv"
survey_lung_cancer = pd.read_csv(file_path)

# Kategorik verileri sayısal verilere dönüştürme
survey_lung_cancer['GENDER'] = survey_lung_cancer['GENDER'].map({'M': 1, 'F': 0})
survey_lung_cancer['LUNG_CANCER'] = survey_lung_cancer['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

# Veri setinin ilk birkaç satırını ve istatistiksel bilgilerini görüntüleme
print("----------ÖNCELİKLE VERİLERİMİZİN STANDART SAPMALARINI VE MATEMATİKSEL FONKSİYONLARIN DEĞERLERİNİ YAZALIM-------------")
print("************************************************************************************************************************")
print(survey_lung_cancer.describe())

print("-------------------------------------------------------------------------------------------------------------------------------------------------------")
print("---------------------------VERİ SETİNDE DAĞILIMINI VE İLİŞKİLERİNİ GÖSTEREN HİSTOGRAM GRAFİKLERİNİ ÇİZDİRELİM VE GÖRSELLEŞTİRELİM-------------------------------")
# Veri setinin dağılımını ve ilişkilerini görselleştirme
sns.pairplot(survey_lung_cancer, diag_kind='kde')
plt.show()

print("------------------------------------------------------------------- KORELASYON MATRİSİNİ HESAPLAYALIM-------------------------------------------------------------------------")
print("*********************************************************************************************************************************************************************************")
print("---------------------------------------------KORELASYON MATRİSİNİ ISI HARİTASI OLARAK GÖRESELLEŞTİRME İŞLEMİ YAPIYORUZ----------------------------------------------------------")
print("Korelasyon matrisi, bir veri setindeki değişkenler arasındaki ilişkileri analiz etmek ve görselleştirmek için kullanılan bir araçtır. Her bir hücre, iki değişken arasındaki korelasyon katsayısını (r) temsil eder. Korelasyon katsayısı,")
print("iki değişken arasındaki doğrusal ilişkiyi ölçer ve -1 ile 1 arasında değişir.")
correlation_matrix = survey_lung_cancer.corr().round(2)
plt.figure(figsize=(16, 12))
sns.heatmap(data=correlation_matrix, annot=True)
plt.title("Correlation Matrix of survey_lung_cancer")
plt.show()

# Veri setini özellikler (X) ve hedef değişken (y) olarak ayırma
X = survey_lung_cancer.drop('LUNG_CANCER', axis=1).values
y = survey_lung_cancer['LUNG_CANCER'].values

# Veriyi %20 test setine ve %80 eğitim setine böleme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Ölçeklendirme, makine öğrenimi algoritmalarının daha iyi performans göstermesine yardımcı olur,
#çünkü özellikler aynı ölçeklerde olduğunda modelin optimizasyon algoritmaları daha kararlı ve hızlı çalışır.
# Özellikle, mesafe tabanlı algoritmalar (örneğin, k-NN, SVM) ve gradient descent tabanlı
#optimizasyon algoritmaları (örneğin, yapay sinir ağları) için ölçeklendirme kritik bir adımdır.
# Verileri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression modelini eğitme
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# SVM modelini eğitme
svm_model = SVR(kernel='linear')
svm_model.fit(X_train_scaled, y_train)

# Random Forest modelini eğitme
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# ANN modellerini tanımlama
class ANN(nn.Module):
    def __init__(self, input_dim):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ANN2(nn.Module):
    def __init__(self, input_dim):
        super(ANN2, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ANN3(nn.Module):
    def __init__(self, input_dim):
        super(ANN3, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.leaky_relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 64)
        self.leaky_relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        x = self.fc2(x)
        x = self.leaky_relu2(x)
        x = self.fc3(x)
        return x

# ANN modellerini eğitme
#X_train.shape[1] ile giriş katmanındaki nöron sayısı, yani özelliklerin (features) sayısı belirlenir.
ann_model = ANN(X_train.shape[1])
#nn.MSELoss(), Ortalama Kare Hatası (Mean Squared Error) kayıp fonksiyonunu tanımlar
#Bu fonksiyon, modelin çıktıları ile gerçek değerler arasındaki hatayı hesaplar.
criterion = nn.MSELoss()

#optim.Adam fonksiyonu, Adam optimizasyon algoritmasını
#kullanarak modelin parametrelerini güncellemek için kullanılır.
optimizer = optim.Adam(ann_model.parameters(), lr=0.001)

for epoch in range(1000):
  #X_train_scaled ve y_train verileri PyTorch tensorlarına dönüştürülür.
    inputs = torch.tensor(X_train_scaled, dtype=torch.float32)
 #, y_train verilerini bir sütun vektörü haline getirir.
    targets = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    optimizer.zero_grad()
    outputs = ann_model(inputs)
    # Tahmin edilen değerler ile gerçek değerler arasındaki kayıp hesaplanır.
    loss = criterion(outputs, targets)
    loss.backward()
    #Adam optimizasyon algoritması kullanılarak parametreler güncellenir.
    optimizer.step()

# İkinci ANN modelini eğitme
ann_model2 = ANN2(X_train.shape[1])
optimizer2 = optim.Adam(ann_model2.parameters(), lr=0.001)

for epoch in range(1000):
    inputs = torch.tensor(X_train_scaled, dtype=torch.float32)
    targets = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    optimizer2.zero_grad()
    outputs = ann_model2(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer2.step()

# Üçüncü ANN modelini eğitme
ann_model3 = ANN3(X_train.shape[1])
optimizer3 = optim.Adam(ann_model3.parameters(), lr=0.1)

for epoch in range(1000):
    inputs = torch.tensor(X_train_scaled, dtype=torch.float32)
    targets = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    optimizer3.zero_grad()
    outputs = ann_model3(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer3.step()

# Linear Regression, SVM, Random Forest ve ANN modellerini değerlendirme
# R² Skoru: Modelin açıklayıcı gücünü ölçer. 1.0 mükemmel uyumu,
# 0.0 modelin hiçbir açıklayıcı gücü olmadığını gösterir.


lr_score = lr_model.score(X_test_scaled, y_test)
svm_score = svm_model.score(X_test_scaled, y_test)
rf_score = rf_model.score(X_test_scaled, y_test)
#çeşitli modellerin performansını test veri seti üzerinde değerlendirir ve her bir modelin R² skorunu hesaplar.
# R² skoru, modelin hedef değişkenin varyansını ne kadar iyi açıkladığını gösteren bir metriktir.
# Bu süreç, modellerin genel performansını karşılaştırmak için kullanılır.
inputs = torch.tensor(X_test_scaled, dtype=torch.float32)
targets = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

ann_outputs = ann_model(inputs)
ann_score = 1 - criterion(ann_outputs, targets).item() / torch.var(targets).item()

ann_outputs2 = ann_model2(inputs)
ann_score2 = 1 - criterion(ann_outputs2, targets).item() / torch.var(targets).item()

ann_outputs3 = ann_model3(inputs)
ann_score3 = 1 - criterion(ann_outputs3, targets).item() / torch.var(targets).item()

# Modellerin performanslarını yazdırma
#print("Linear Regression Score:", lr_score)
#print("SVM Score:", svm_score)
#print("Random Forest Score:", rf_score)
#print("ANN Score (Model 1):", ann_score)
#print("ANN2 Score (Model 2):", ann_score2)
#print("ANN3 Score (Model 3):", ann_score3)

# K-Nearest Neighbors Classifier
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)
knn_conf = confusion_matrix(y_test, knn_pred)
knn_report = classification_report(y_test, knn_pred)
knn_acc = round(accuracy_score(y_test, knn_pred) * 100, ndigits=2)
print(f"\nThe Accuracy of K Nearest Neighbors Classifier is {knn_acc} %")





import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

# Veri setini yükleme
file_path = "/content/drive/MyDrive/survey lung cancer.csv"
survey_lung_cancer = pd.read_csv(file_path)

# Kategorik verileri sayısal verilere dönüştürme
survey_lung_cancer['GENDER'] = survey_lung_cancer['GENDER'].map({'M': 1, 'F': 0})
survey_lung_cancer['LUNG_CANCER'] = survey_lung_cancer['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

# Veri setinin ilk birkaç satırını ve istatistiksel bilgilerini görüntüleme
print("----------ÖNCELİKLE VERİLERİMİZİN STANDART SAPMALARINI VE MATEMATİKSEL FONKSİYONLARIN DEĞERLERİNİ YAZALIM-------------")
print("************************************************************************************************************************")
#print(survey_lung_cancer.describe())

print("-------------------------------------------------------------------------------------------------------------------------------------------------------")
print("---------------------------VERİ SETİNDE DAĞILIMINI VE İLİŞKİLERİNİ GÖSTEREN HİSTOGRAM GRAFİKLERİNİ ÇİZDİRELİM VE GÖRSELLEŞTİRELİM-------------------------------")
# Veri setinin dağılımını ve ilişkilerini görselleştirme
sns.pairplot(survey_lung_cancer, diag_kind='kde')
plt.show()

print("------------------------------------------------------------------- KORELASYON MATRİSİNİ HESAPLAYALIM-------------------------------------------------------------------------")
print("*********************************************************************************************************************************************************************************")
print("---------------------------------------------KORELASYON MATRİSİNİ ISI HARİTASI OLARAK GÖRESELLEŞTİRME İŞLEMİ YAPIYORUZ----------------------------------------------------------")
correlation_matrix = survey_lung_cancer.corr().round(2)
plt.figure(figsize=(16, 12))
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of survey_lung_cancer")
plt.show()

# Veri setini özellikler (X) ve hedef değişken (y) olarak ayırma
X = survey_lung_cancer.drop('LUNG_CANCER', axis=1).values
y = survey_lung_cancer['LUNG_CANCER'].values

# Veriyi %20 test setine ve %80 eğitim setine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verileri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression modelini eğitme
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# SVM modelini eğitme
svm_model = SVR(kernel='linear')
svm_model.fit(X_train_scaled, y_train)

# Random Forest modelini eğitme
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# ANN modellerini tanımlama
class ANN(nn.Module):
    def __init__(self, input_dim):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ANN2(nn.Module):
    def __init__(self, input_dim):
        super(ANN2, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ANN3(nn.Module):
    def __init__(self, input_dim):
        super(ANN3, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.leaky_relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 64)
        self.leaky_relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        x = self.fc2(x)
        x = self.leaky_relu2(x)
        x = self.fc3(x)
        return x

# ANN modellerini eğitme
ann_model = ANN(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(ann_model.parameters(), lr=0.001)

for epoch in range(1000):
    inputs = torch.tensor(X_train_scaled, dtype=torch.float32)
    targets = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    optimizer.zero_grad()
    outputs = ann_model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

# İkinci ANN modelini eğitme
ann_model2 = ANN2(X_train.shape[1])
optimizer2 = optim.Adam(ann_model2.parameters(), lr=0.001)

for epoch in range(1000):
    inputs = torch.tensor(X_train_scaled, dtype=torch.float32)
    targets = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    optimizer2.zero_grad()
    outputs = ann_model2(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer2.step()

# Üçüncü ANN modelini eğitme
ann_model3 = ANN3(X_train.shape[1])
optimizer3 = optim.Adam(ann_model3.parameters(), lr=0.1)

for epoch in range(1000):
    inputs = torch.tensor(X_train_scaled, dtype=torch.float32)
    targets = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    optimizer3.zero_grad()
    outputs = ann_model3(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer3.step()

# Linear Regression, SVM, Random Forest ve ANN modellerini değerlendirme
lr_score = lr_model.score(X_test_scaled, y_test)
svm_score = svm_model.score(X_test_scaled, y_test)
rf_score = rf_model.score(X_test_scaled, y_test)

inputs = torch.tensor(X_test_scaled, dtype=torch.float32)
targets = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

ann_outputs = ann_model(inputs)
ann_score = 1 - criterion(ann_outputs, targets).item() / torch.var(targets).item()

ann_outputs2 = ann_model2(inputs)
ann_score2 = 1 - criterion(ann_outputs2, targets).item() / torch.var(targets).item()

ann_outputs3 = ann_model3(inputs)
ann_score3 = 1 - criterion(ann_outputs3, targets).item() / torch.var(targets).item()

# Modellerin performanslarını yazdırma
#print("Linear Regression Score:", lr_score)
#print("SVM Score:", svm_score)
#print("Random Forest Score:", rf_score)
#print("ANN Score (Model 1):", ann_score)
#print("ANN2 Score (Model 2):", ann_score2)
#print("ANN3 Score (Model 3):", ann_score3)

# K-Nearest Neighbors Classifier
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)
knn_conf = confusion_matrix(y_test, knn_pred)
knn_report = classification_report(y_test, knn_pred)
knn_acc = round(accuracy_score(y_test, knn_pred) * 100, ndigits=2)
#print(f"\nThe Accuracy of K Nearest Neighbors Classifier is {knn_acc} %")



print("KERAS  KÜTÜPHANESİ   DENEYELİM  KERAS  KÜTÜPHANESİYLE  ALDIĞIMIZ SONUÇ  EN ALTTA YAZIYOR ")


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("------------------------------ŞİMDİ DE KERAS KÜTÜPHANESİYLE DENEYELİM---------------------------")

# Dosya yolunu belirtin ve veri setini yükleyin
file_path = "/content/drive/MyDrive/survey lung cancer.csv"
data = pd.read_csv(file_path)

# Kategorik verileri sayısal verilere dönüştürün
data = pd.get_dummies(data, drop_first=True)

# Özellikler (X) ve hedef değişken (y) olarak ayırın
x = data.drop("LUNG_CANCER_YES", axis=1)  # Dönüştürülmüş kategorik veriden gelen yeni sütun ismiyle uyumlu olmalı
y = data["LUNG_CANCER_YES"]

# Özellikleri ölçeklendirin
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Veriyi eğitim ve test setlerine ayırın
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Hedef değişkeni float32 türüne dönüştürün
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# Nöral ağı tanımlayın
#Burada regularization_parameter adı verilen bir değer kullanılır. Bu değer, ağırlık düzenleme terimini
#kontrol eder. Daha yüksek bir regularization_parameter değeri, ağırlıkların düzenlenmesinde daha
#fazla kısıtlama sağlar, bu da daha fazla basitleşmeye ve aşırı uymayı azaltmaya yol açabilir.
#Ancak, bu değerin doğru bir şekilde seçilmesi, aşırı uyumu kontrol etmek için önemlidir
regularization_parameter = 0.003
#Dense; katmanları,;;;;;;; tam bağlı (fully connected) katmanları temsil eder. Burada her bir nöron, önceki katmandaki her bir nörondan giriş alır
#units parametresi, ;;;;katmandaki nöron sayısını belirtir. Örneğin, ilk katmanda 32 nöron, ikinci katmanda 64 nöron ve üçüncü katmanda 128 nöron bulunmaktadır.
#input_dim parametresi,;;;;; ilk katman için giriş boyutunu belirtir. Bu durumda, giriş boyutu x_train.shape[-1] olup, x_train veri setinin son boyutu (özellik sayısı) alınır.
#kernel_regularizer parametresi;;;;, ağırlık düzenleme (regularization) yöntemini belirtir. Burada, regularizers.l1(regularization_parameter) şeklinde belirtilmiş ve L1 düzenleme
#kullanılmıştır. L1 düzenleme, ağırlıkları düzenlemek için L1 normunu kullanır. Bu, ağırlıkların büyüklüğünü kontrol ederek aşırı uymayı azaltmaya yardımcı olur.
#Dropout katmanı;;;;, ağın aşırı uyumu önlemek için kullanılan bir regülerleştirme tekniğidir. Belirtilen olasılık (0.3) ile her bir eğitim adımında rastgele seçilen
#nöronların belirtilen oranda atılmasıdır. Bu, ağın genelleme yapmasını teşvik eder.

neural_model = Sequential([

    Dense(units=32, input_dim=x_train.shape[-1], activation="relu", kernel_regularizer=regularizers.l1(regularization_parameter)),


    Dropout(0.3),
    Dense(units=16, activation="relu", kernel_regularizer=regularizers.l1(regularization_parameter)),
    Dense(units=1, activation="sigmoid")
])

# Model özetini yazdırın
print(neural_model.summary())


print("------------------------------------KATMANLARDA BULUNAN NORON SAYISINI EKRANA YAZDIRALIM -------------------------------------")

print("--------------------------------TOTALDE KAÇ TANE PARAMETRE VAR  BUNLARI DA EKRANA YAZDIRALIM----------------------------------")


print("-----------------------------------------EN SONDA  EĞİTİLMEYEN NORONLARI YAZDIRALIIM ------------------------------------------- ")
# Modeli derleyin
neural_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

#EPOK SAYISINI  VE MODELİ BURDA EĞİTİYORUZ

# Modeli eğitin
history = neural_model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Modeli değerlendirin
loss, accuracy = neural_model.evaluate(x_test, y_test)
print(f'ANN WİTH KERASTest Loss: {loss}')
print(f'ANN WİTH KERAS Test Accuracy: {accuracy}')



print("LİNEER REGRESYON MODELİNİ BURDA EĞİTİP BAŞARI RAPORUNU YAZDIRIYORUZ")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#regresyon yapılıyor burda
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)
lr_conf = confusion_matrix(y_test, lr_pred)
lr_report = classification_report(y_test, lr_pred)
lr_acc = round(accuracy_score(y_test, lr_pred)*100, ndigits = 2)
#print(f"Confusion Matrix : \n\n{lr_conf}")
#print(f"\nClassification Report : \n\n{lr_report}")
#print(f"\nThe Accuracy of Logistic Regression is {lr_acc} %")



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)
knn_conf = confusion_matrix(y_test, knn_pred)
knn_report = classification_report(y_test, knn_pred)
knn_acc = round(accuracy_score(y_test, knn_pred)*100, ndigits = 2)
#Bu satırda örneğin Normalde  0 ile 1 arasında değer alır  bunu da yüzdeye vuruyor
#print(f"Confusion Matrix : \n\n{knn_conf}")
#print(f"\nClassification Report : \n\n{knn_report}")
#print(f"\nThe Accuracy of K Nearest Neighbors Classifier is {knn_acc} %")
print("KNN ALGORİTMASINI DA  BURDA KULLANMIŞ OLDUK  HER MODELDE  X VE  Y KENDİ İÇİNDE TANIMLIYORUZ ")



print("Linear Regression Score:", lr_score)
print("SVM Score:", svm_score)
print("Random Forest Score:", rf_score)
print("ANN(PYTORCH)Score (Model 1):", ann_score)
print("ANN(PYTORCH)Score (Model 2):", ann_score2)

#print(f'ANN WİTH KERAS Test Loss: {loss}')
print(f'ANN(KERAS) Accuracy: {accuracy}')
print(f"KNN Classifier Accuracy{knn_acc} %")
print(f"Logistic Regression Accuracy  {lr_acc} %")


print("Bu kod parçası, bir modelin eğitim ve doğrulama süreçlerindeki doğruluk ve kayıp metriklerini görselleştirerek, modelin performansını zamanla izlemeyi sağlar.")
print("Bu tür grafikler, modelin aşırı öğrenme (overfitting) veya yetersiz öğrenme (underfitting) gibi problemleri olup olmadığını anlamaya yardımcı olur. Eğer")
print("eğitim doğruluğu sürekli artarken doğrulama doğruluğu bir noktadan sonra sabit kalıyor veya düşüyorsa, bu aşırı öğrenmenin bir göstergesi olabilir")

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(len(acc)) # number of epochs

plt.figure(figsize=(20, 12))
plt.subplot(2,1,1)
plt.plot(epochs, acc, "yellow", label= "Training Accuracy")
plt.plot(epochs, val_acc, "black", label= "Validation Accuracy")
plt.title("Training and validation accuracy")
plt.legend()

plt.subplot(2,1,2)
plt.plot(epochs, loss, "yellow", label= "Training Loss")
plt.plot(epochs, val_loss, "black", label= "Validation Loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()



print("KERAS  KÜTÜPHANESİ   DENEYELİM  KERAS  KÜTÜPHANESİYLE  ALDIĞIMIZ SONUÇ  EN ALTTA YAZIYOR ")


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("------------------------------ŞİMDİ DE KERAS KÜTÜPHANESİYLE DENEYELİM---------------------------")

# Dosya yolunu belirtin ve veri setini yükleyin
file_path = "/content/drive/MyDrive/survey lung cancer.csv"
data = pd.read_csv(file_path)

# Kategorik verileri sayısal verilere dönüştürün
data = pd.get_dummies(data, drop_first=True)

# Özellikler (X) ve hedef değişken (y) olarak ayırın
x = data.drop("LUNG_CANCER_YES", axis=1)  # Dönüştürülmüş kategorik veriden gelen yeni sütun ismiyle uyumlu olmalı
y = data["LUNG_CANCER_YES"]

# Özellikleri ölçeklendirin
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Veriyi eğitim ve test setlerine ayırın
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Hedef değişkeni float32 türüne dönüştürün
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# Nöral ağı tanımlayın
regularization_parameter = 0.003
neural_model = Sequential([
    Dense(units=63, input_dim=x_train.shape[-1], activation="LeakyReLU", kernel_regularizer=regularizers.l1(regularization_parameter)),
    Dense(units=32, activation="LeakyReLU", kernel_regularizer=regularizers.l1(regularization_parameter)),
    Dense(units=16, activation="LeakyReLU", kernel_regularizer=regularizers.l1(regularization_parameter)),
    Dense(units=16, activation="LeakyReLU", kernel_regularizer=regularizers.l1(regularization_parameter)),
    Dense(units=1, activation="LeakyReLU")
])

# Model özetini yazdırın
print(neural_model.summary())


print("------------------------------------KATMANLARDA BULUNAN NORON SAYISINI EKRANA YAZDIRALIM -------------------------------------")

print("--------------------------------TOTALDE KAÇ TANE PARAMETRE VAR  BUNLARI DA EKRANA YAZDIRALIM----------------------------------")


print("-----------------------------------------EN SONDA  EĞİTİLMEYEN NORONLARI YAZDIRALIIM ------------------------------------------- ")
# Modeli derleyin
neural_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

#EPOK SAYISINI  VE MODELİ BURDA EĞİTİYORUZ

# Modeli eğitin
history = neural_model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2)

# Modeli değerlendirin
loss, accuracy = neural_model.evaluate(x_test, y_test)
print(f'ANN WİTH KERASTest Loss: {loss}')
print(f'ANN WİTH KERAS Test Accuracy: {accuracy}')

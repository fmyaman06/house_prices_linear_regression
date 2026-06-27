import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. VERİ YÜKLEME 
df = pd.read_csv('data.csv')
df = df[df['price'] > 0]

# 2.  ENCODE 
city_mean = df.groupby('city')['price'].mean()
df['city_encoded'] = df['city'].map(city_mean)

# 3. FEATURE SEÇİMİ
features = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
    'floors', 'waterfront', 'view', 'condition',
    'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
    'city_encoded'
]

X = df[features].fillna(df[features].median())
y = df['price']

#
# 4. TRAIN / TEST SPLIT 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  5. PİPELINE TANIMLA 
# Her model için: StandardScaler  Model
pipelines = {
    'Linear Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LinearRegression())
    ]),
    'Ridge (alpha=1)': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  Ridge(alpha=1))
    ]),
    'Ridge (alpha=10)': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  Ridge(alpha=10))
    ]),
    'Ridge (alpha=100)': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  Ridge(alpha=100))
    ]),
    'Lasso (alpha=1)': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  Lasso(alpha=1, max_iter=10000))
    ]),
    'Lasso (alpha=10)': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  Lasso(alpha=10, max_iter=10000))
    ]),
    'Lasso (alpha=100)': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  Lasso(alpha=100, max_iter=10000))
    ]),
}

# 6. TÜM PİPELINE'LARI EĞİT VE KARŞILAŞTIR
results = []

for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)          
    y_pred = pipe.predict(X_test)

    rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
    r2    = r2_score(y_test, y_pred)
    cv_r2 = cross_val_score(pipe, X_train, y_train,
                             cv=5, scoring='r2').mean()

    results.append({
        'Model' : name,
        'RMSE'  : round(rmse, 0),
        'R²'    : round(r2, 4),
        'CV R²' : round(cv_r2, 4)
    })

results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
print(results_df.to_string(index=False))

#  7. EN İYİ MODELİ SEÇ 
best_name = results_df.iloc[0]['Model']
best_pipe = pipelines[best_name]
print(f"\n✅ En iyi model: {best_name}")

#  8. GRAFİKLER 
# Grafik 1: R² Karşılaştırma
colors = ['steelblue' if 'Linear' in m else
          'coral'     if 'Ridge'  in m else
          'mediumseagreen' for m in results_df['Model']]

plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x='R²', y='Model', palette=colors)
plt.title('Pipeline Karşılaştırması — R² Skoru')
plt.axvline(0.7, color='red', linestyle='--', label='0.7 eşiği')
plt.xlim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()

# Grafik 2: RMSE Karşılaştırma
plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x='RMSE', y='Model', palette=colors)
plt.title('Pipeline Karşılaştırması — RMSE (düşük = iyi)')
plt.tight_layout()
plt.show()

# Grafik 3: En iyi modelin Gerçek vs Tahmin
best_pred = best_pipe.predict(X_test)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, best_pred, alpha=0.4, color='steelblue')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Gerçek Fiyat')
plt.ylabel('Tahmin Edilen Fiyat')
plt.title(f'{best_name} — Gerçek vs Tahmin')
plt.tight_layout()
plt.show()

# Grafik 4: Alpha'ya göre R² değişimi
alphas = [0.01, 0.1, 1, 10, 50, 100, 500, 1000]
ridge_scores, lasso_scores = [], []

for a in alphas:
    r_pipe = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=a))])
    l_pipe = Pipeline([('scaler', StandardScaler()), ('model', Lasso(alpha=a, max_iter=10000))])

    r_pipe.fit(X_train, y_train)
    l_pipe.fit(X_train, y_train)

    ridge_scores.append(r2_score(y_test, r_pipe.predict(X_test)))
    lasso_scores.append(r2_score(y_test, l_pipe.predict(X_test)))

plt.figure(figsize=(10, 5))
plt.plot(alphas, ridge_scores, marker='o', label='Ridge', color='coral')
plt.plot(alphas, lasso_scores, marker='s', label='Lasso', color='mediumseagreen')
plt.xscale('log')
plt.xlabel('Alpha (log scale)')
plt.ylabel('R²')
plt.title('Alpha Değerine Göre R² Değişimi')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
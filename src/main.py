# =============================================================================
# PROJETO INDIVIDUAL DE MACHINE LEARNING - PI1
# PREDIÇÃO DE PREÇOS DE IMÓVEIS COM DATASET REAL
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


# Configuração de visualização - CORRIGIDO
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('ggplot')
        
sns.set_palette("husl")

# =============================================================================
# 1. DESCRIÇÃO DO PROBLEMA
# =============================================================================
"""
PROBLEMA: Prever o preço de imóveis (em USD) com base em suas características
como tipo de propriedade, localização, tamanho, etc.

OBJETIVO: Desenvolver modelos de machine learning que possam estimar com 
precisão o valor dos imóveis usando as features disponíveis no dataset.
"""

print("=" * 70)
print("PROJETO DE MACHINE LEARNING - PREVISÃO DE PREÇOS DE IMÓVEIS")
print("DATASET REAL")
print("=" * 70)

# =============================================================================
# 2. ETL E LIMPEZA DOS DADOS
# =============================================================================
print("\n2. ETL E LIMPEZA DOS DADOS")

# CARREGUE SEU DATASET REAL AQUI
df = pd.read_csv('data\processed\housing_data_CDMX_cleaned.csv')


# ANÁLISE EXPLORATÓRIA INICIAL
print("\n" + "="*50)
print("ANÁLISE EXPLORATÓRIA INICIAL")
print("="*50)

print("\nPrimeiras 5 linhas do dataset:")
print(df.head())

print(f"\nShape do dataset: {df.shape}")
print(f"\nColunas disponíveis: {list(df.columns)}")

print("\nInformações do dataset:")
print(df.info())

print("\nEstatísticas descritivas das variáveis numéricas:")
print(df.describe())

print("\nValores nulos por coluna:")
print(df.isnull().sum())

# ANALISAR COLUNAS CATEGÓRICAS DISPONÍVEIS
categorical_columns = df.select_dtypes(include=['object']).columns
print(f"\nColunas categóricas: {list(categorical_columns)}")

for col in categorical_columns:
    print(f"\n{col}:")
    print(df[col].value_counts().head())

# LIMPEZA E PREPARAÇÃO DOS DADOS
print("\n" + "="*50)
print("LIMPEZA E PREPARAÇÃO DOS DADOS")
print("="*50)

# Verificar e tratar valores nulos (se houver)
if df.isnull().sum().sum() > 0:
    print("Tratando valores nulos...")
    # Para variáveis numéricas
    num_imputer = SimpleImputer(strategy='median')
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    
    # Para variáveis categóricas
    cat_imputer = SimpleImputer(strategy='most_frequent')
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
else:
    print("Nenhum valor nulo encontrado.")

# Engenharia de features
print("\nCriando novas features...")

# Verificar quais colunas de área estão disponíveis
area_columns = [col for col in df.columns if 'surface' in col.lower() or 'area' in col.lower() or 'm2' in col]
print(f"Colunas de área disponíveis: {area_columns}")

if 'surface_total_in_m2' in df.columns and 'surface_covered_in_m2' in df.columns:
    # Feature: diferença entre área total e área coberta
    df['area_difference'] = df['surface_total_in_m2'] - df['surface_covered_in_m2']
    # Feature: ratio área coberta/total
    df['area_ratio'] = df['surface_covered_in_m2'] / df['surface_total_in_m2']
    print("Features de área criadas: area_difference, area_ratio")
else:
    print("Colunas de área não encontradas para criar features adicionais")

# Codificar variáveis categóricas
print("\nCodificando variáveis categóricas...")
label_encoders = {}

# Selecionar colunas categóricas para codificar (excluir colunas com muitos valores únicos)
categorical_to_encode = []
for col in categorical_columns:
    if df[col].nunique() <= 20:  # Codificar apenas colunas com até 20 categorias
        categorical_to_encode.append(col)
    else:
        print(f"Coluna '{col}' tem {df[col].nunique()} categorias - não será codificada")

print(f"Colunas a serem codificadas: {categorical_to_encode}")

for column in categorical_to_encode:
    le = LabelEncoder()
    df[column + '_encoded'] = le.fit_transform(df[column])
    label_encoders[column] = le
    print(f"{column}: {len(le.classes_)} categorias")

# =============================================================================
# 3. GRÁFICOS E VISUALIZAÇÕES
# =============================================================================
print("\n3. ANÁLISE EXPLORATÓRIA E VISUALIZAÇÕES")

# Configuração dos gráficos - ADAPTATIVO
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('Análise Exploratória - Dataset de Imóveis', fontsize=16, fontweight='bold')

# 1. Distribuição do preço em USD (usar a primeira coluna de preço disponível)
price_column = 'price_aprox_usd' if 'price_aprox_usd' in df.columns else 'price'
axes[0,0].hist(df[price_column], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].set_title(f'Distribuição do Preço ({price_column})')
axes[0,0].set_xlabel('Preço')
axes[0,0].set_ylabel('Frequência')

# 2. Distribuição da área total (se disponível)
if 'surface_total_in_m2' in df.columns:
    axes[0,1].hist(df['surface_total_in_m2'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0,1].set_title('Distribuição da Área Total (m²)')
    axes[0,1].set_xlabel('Área Total (m²)')
    axes[0,1].set_ylabel('Frequência')
else:
    axes[0,1].text(0.5, 0.5, 'Área total não disponível', ha='center', va='center', transform=axes[0,1].transAxes)
    axes[0,1].set_title('Área Total - Dados não disponíveis')

# 3. Preço por tipo de propriedade (se disponível)
if 'property_type' in df.columns:
    df.boxplot(column=price_column, by='property_type', ax=axes[0,2])
    axes[0,2].set_title('Preço por Tipo de Propriedade')
    axes[0,2].set_xlabel('Tipo de Propriedade')
    axes[0,2].set_ylabel('Preço')
else:
    axes[0,2].text(0.5, 0.5, 'Tipo de propriedade não disponível', ha='center', va='center', transform=axes[0,2].transAxes)
    axes[0,2].set_title('Preço por Tipo - Dados não disponíveis')

# 4. Relação área total vs preço (se disponível)
if 'surface_total_in_m2' in df.columns:
    axes[1,0].scatter(df['surface_total_in_m2'], df[price_column], alpha=0.6, color='green')
    axes[1,0].set_title('Área Total vs Preço')
    axes[1,0].set_xlabel('Área Total (m²)')
    axes[1,0].set_ylabel('Preço')
else:
    # Usar outra variável numérica se área não estiver disponível
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        other_numeric = [col for col in numeric_cols if col != price_column][0]
        axes[1,0].scatter(df[other_numeric], df[price_column], alpha=0.6, color='green')
        axes[1,0].set_title(f'{other_numeric} vs Preço')
        axes[1,0].set_xlabel(other_numeric)
        axes[1,0].set_ylabel('Preço')
    else:
        axes[1,0].text(0.5, 0.5, 'Dados numéricos insuficientes', ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('Relação com Preço - Dados não disponíveis')

# 5. Preço por m² por tipo de propriedade (se disponível)
if 'price_usd_per_m2' in df.columns and 'property_type' in df.columns:
    df.boxplot(column='price_usd_per_m2', by='property_type', ax=axes[1,1])
    axes[1,1].set_title('Preço por m² por Tipo de Propriedade')
    axes[1,1].set_xlabel('Tipo de Propriedade')
    axes[1,1].set_ylabel('Preço por m²')
else:
    axes[1,1].text(0.5, 0.5, 'Dados de preço por m² não disponíveis', ha='center', va='center', transform=axes[1,1].transAxes)
    axes[1,1].set_title('Preço por m² - Dados não disponíveis')

# 6. Mapa de calor de correlações
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 1:
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, ax=axes[1,2])
    axes[1,2].set_title('Matriz de Correlação')
else:
    axes[1,2].text(0.5, 0.5, 'Dados insuficientes\npara correlação', ha='center', va='center', transform=axes[1,2].transAxes)
    axes[1,2].set_title('Matriz de Correlação - Dados não disponíveis')

# 7. Distribuição geográfica (se coordenadas disponíveis)
if 'latitude' in df.columns and 'longitude' in df.columns:
    scatter = axes[2,0].scatter(df['longitude'], df['latitude'], c=df[price_column], 
                               cmap='viridis', alpha=0.6, s=10)
    axes[2,0].set_title('Distribuição Geográfica - Preços')
    axes[2,0].set_xlabel('Longitude')
    axes[2,0].set_ylabel('Latitude')
    plt.colorbar(scatter, ax=axes[2,0], label='Preço')
else:
    axes[2,0].text(0.5, 0.5, 'Coordenadas geográficas\nnão disponíveis', ha='center', va='center', transform=axes[2,0].transAxes)
    axes[2,0].set_title('Distribuição Geográfica - Dados não disponíveis')

# 8. Distribuição do preço por m² (se disponível)
if 'price_usd_per_m2' in df.columns:
    axes[2,1].hist(df['price_usd_per_m2'], bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[2,1].set_title('Distribuição do Preço por m²')
    axes[2,1].set_xlabel('Preço por m²')
    axes[2,1].set_ylabel('Frequência')
else:
    axes[2,1].text(0.5, 0.5, 'Preço por m² não disponível', ha='center', va='center', transform=axes[2,1].transAxes)
    axes[2,1].set_title('Preço por m² - Dados não disponíveis')

# 9. Top locais por preço médio (se disponível)
if 'places' in df.columns:
    top_places = df.groupby('places')[price_column].mean().sort_values(ascending=False).head(10)
    top_places.plot(kind='bar', ax=axes[2,2], color='purple', alpha=0.7)
    axes[2,2].set_title('Top Locais - Preço Médio')
    axes[2,2].set_xlabel('Localização')
    axes[2,2].set_ylabel('Preço Médio')
    axes[2,2].tick_params(axis='x', rotation=45)
else:
    axes[2,2].text(0.5, 0.5, 'Dados de localização\nnão disponíveis', ha='center', va='center', transform=axes[2,2].transAxes)
    axes[2,2].set_title('Locais - Dados não disponíveis')

plt.tight_layout()
plt.show()

# Gráficos adicionais
print("\nGráficos adicionais de análise...")

# Análise de outliers
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Boxplot do preço
ax1.boxplot(df[price_column])
ax1.set_title(f'Boxplot - {price_column}')
ax1.set_ylabel('Preço')

# Boxplot do preço por m² (se disponível)
if 'price_usd_per_m2' in df.columns:
    ax2.boxplot(df['price_usd_per_m2'])
    ax2.set_title('Boxplot - Preço por m²')
    ax2.set_ylabel('Preço por m²')
else:
    ax2.text(0.5, 0.5, 'Preço por m²\nnão disponível', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Preço por m² - Dados não disponíveis')

plt.tight_layout()
plt.show()

# =============================================================================
# PREPARAÇÃO DOS DADOS PARA MODELAGEM
# =============================================================================
print("\nPREPARAÇÃO DOS DADOS PARA MODELAGEM")

# Definir features disponíveis dinamicamente
print("\nSelecionando features para o modelo...")

# Lista de possíveis features numéricas
possible_numeric_features = [
    'surface_total_in_m2', 'surface_covered_in_m2', 'price_usd_per_m2',
    'latitude', 'longitude', 'area_difference', 'area_ratio'
]

# Lista de possíveis features categóricas codificadas
possible_categorical_features = [col for col in df.columns if col.endswith('_encoded')]

# Selecionar apenas as features que existem no dataset
available_features = []
for feature in possible_numeric_features + possible_categorical_features:
    if feature in df.columns:
        available_features.append(feature)

print(f"Features disponíveis: {available_features}")

if not available_features:
    print("ERRO: Nenhuma feature disponível para treinamento!")
    exit()

# Definindo features e target
X = df[available_features]
y = df[price_column]

print(f"\nFeatures selecionadas: {len(available_features)}")
print(f"Target: {price_column}")
print(f"Shape de X: {X.shape}")
print(f"Shape de y: {y.shape}")

# Dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\nDivisão treino/teste:")
print(f"Treino: {X_train.shape[0]} amostras ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Teste: {X_test.shape[0]} amostras ({X_test.shape[0]/len(X)*100:.1f}%)")

# Normalizando os dados (apenas para modelos que precisam)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nDados normalizados com StandardScaler")

# =============================================================================
# 4. MODELAGEM E ANÁLISE DOS RESULTADOS
# =============================================================================
print("\n4. MODELAGEM E ANÁLISE DOS RESULTADOS")

# =============================================================================
# MODELO 1: REGRESSÃO LINEAR
# =============================================================================
print("\n" + "="*50)
print("MODELO 1: REGRESSÃO LINEAR")
print("="*50)

# Criando e treinando o modelo
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Fazendo previsões
y_pred_lr = lr_model.predict(X_test_scaled)

# Calculando métricas
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Validação cruzada
cv_scores_lr = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='r2')

print("=== MÉTRICAS DE DESEMPENHO ===")
print(f"MSE: {mse_lr:,.2f}")
print(f"RMSE: {rmse_lr:,.2f}")
print(f"MAE: {mae_lr:,.2f}")
print(f"R²: {r2_lr:.4f}")
print(f"R² Validação Cruzada (5-fold): {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std() * 2:.4f})")

# Coeficientes da regressão
print("\n=== COEFICIENTES DA REGRESSÃO LINEAR ===")
feature_importance_lr = pd.DataFrame({
    'feature': available_features,
    'coefficient': lr_model.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print(feature_importance_lr.head(10))

# =============================================================================
# MODELO 2: RANDOM FOREST
# =============================================================================
print("\n" + "="*50)
print("MODELO 2: RANDOM FOREST")
print("="*50)

# Criando e treinando o modelo
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Fazendo previsões
y_pred_rf = rf_model.predict(X_test)

# Calculando métricas
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Validação cruzada
cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')

print("=== MÉTRICAS DE DESEMPENHO ===")
print(f"MSE: {mse_rf:,.2f}")
print(f"RMSE: {rmse_rf:,.2f}")
print(f"MAE: {mae_rf:,.2f}")
print(f"R²: {r2_rf:.4f}")
print(f"R² Validação Cruzada (5-fold): {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std() * 2:.4f})")

# =============================================================================
# COMPARAÇÃO DOS MODELOS
# =============================================================================
print("\n" + "="*50)
print("COMPARAÇÃO DOS MODELOS")
print("="*50)

comparacao = pd.DataFrame({
    'Métrica': ['RMSE', 'MAE', 'R²', 'R² CV'],
    'Regressão Linear': [
        f"{rmse_lr:,.2f}", 
        f"{mae_lr:,.2f}", 
        f"{r2_lr:.4f}",
        f"{cv_scores_lr.mean():.4f}"
    ],
    'Random Forest': [
        f"{rmse_rf:,.2f}", 
        f"{mae_rf:,.2f}", 
        f"{r2_rf:.4f}",
        f"{cv_scores_rf.mean():.4f}"
    ]
})

print(comparacao)

# =============================================================================
# VISUALIZAÇÃO DOS RESULTADOS
# =============================================================================
print("\nGerando visualizações dos resultados...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Análise Comparativa dos Modelos', fontsize=16, fontweight='bold')

# 1. Previsões vs Valores Reais - Regressão Linear
axes[0,0].scatter(y_test, y_pred_lr, alpha=0.6, color='blue', s=20)
axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0,0].set_title(f'Regressão Linear (R² = {r2_lr:.3f})')
axes[0,0].set_xlabel('Valor Real')
axes[0,0].set_ylabel('Previsão')

# 2. Previsões vs Valores Reais - Random Forest
axes[0,1].scatter(y_test, y_pred_rf, alpha=0.6, color='green', s=20)
axes[0,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0,1].set_title(f'Random Forest (R² = {r2_rf:.3f})')
axes[0,1].set_xlabel('Valor Real')
axes[0,1].set_ylabel('Previsão')

# 3. Importância das variáveis - Random Forest
feature_importance_rf = pd.DataFrame({
    'feature': available_features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

sns.barplot(data=feature_importance_rf.head(10), x='importance', y='feature', ax=axes[0,2])
axes[0,2].set_title('Importância das Variáveis - Random Forest')
axes[0,2].set_xlabel('Importância')

# 4. Resíduos - Regressão Linear
residuos_lr = y_test - y_pred_lr
axes[1,0].scatter(y_pred_lr, residuos_lr, alpha=0.6, color='blue', s=20)
axes[1,0].axhline(y=0, color='red', linestyle='--')
axes[1,0].set_title('Resíduos - Regressão Linear')
axes[1,0].set_xlabel('Previsões')
axes[1,0].set_ylabel('Resíduos')

# 5. Resíduos - Random Forest
residuos_rf = y_test - y_pred_rf
axes[1,1].scatter(y_pred_rf, residuos_rf, alpha=0.6, color='green', s=20)
axes[1,1].axhline(y=0, color='red', linestyle='--')
axes[1,1].set_title('Resíduos - Random Forest')
axes[1,1].set_xlabel('Previsões')
axes[1,1].set_ylabel('Resíduos')

# 6. Distribuição dos erros
axes[1,2].hist(residuos_lr, bins=50, alpha=0.7, color='blue', label='Regressão Linear', density=True)
axes[1,2].hist(residuos_rf, bins=50, alpha=0.7, color='green', label='Random Forest', density=True)
axes[1,2].set_title('Distribuição dos Resíduos')
axes[1,2].set_xlabel('Erro')
axes[1,2].set_ylabel('Densidade')
axes[1,2].legend()

plt.tight_layout()
plt.show()

# =============================================================================
# ANÁLISE DE PERFORMANCE POR CATEGORIA
# =============================================================================
print("\nANÁLISE DE PERFORMANCE POR CATEGORIA")

# Adicionando previsões ao dataset de teste
results_df = X_test.copy()
results_df['price_real'] = y_test.values
results_df['price_pred_lr'] = y_pred_lr
results_df['price_pred_rf'] = y_pred_rf

# Adicionar coluna de tipo de propriedade se disponível
if 'property_type' in df.columns:
    results_df['property_type'] = df.loc[X_test.index, 'property_type'].values

# Calculando erro percentual
results_df['error_pct_lr'] = np.abs(results_df['price_real'] - results_df['price_pred_lr']) / results_df['price_real'] * 100
results_df['error_pct_rf'] = np.abs(results_df['price_real'] - results_df['price_pred_rf']) / results_df['price_real'] * 100

# Performance por tipo de propriedade (se disponível)
if 'property_type' in results_df.columns:
    performance_by_type = results_df.groupby('property_type').agg({
        'error_pct_lr': 'mean',
        'error_pct_rf': 'mean',
        'price_real': 'count'
    }).round(2)

    performance_by_type.columns = ['Erro Médio LR (%)', 'Erro Médio RF (%)', 'Número de Amostras']
    print("\nPerformance por Tipo de Propriedade:")
    print(performance_by_type)
else:
    print("\nColuna 'property_type' não disponível para análise de performance por categoria")

# =============================================================================
# INTERPRETAÇÃO DOS RESULTADOS
# =============================================================================
print("\n" + "="*70)
print("INTERPRETAÇÃO DOS RESULTADOS")
print("="*70)

print(f"""
ANÁLISE COMPARATIVA DOS MODELOS:

1. DESEMPENHO GERAL:
   - Regressão Linear: R² = {r2_lr:.3f}, RMSE = {rmse_lr:,.0f}
   - Random Forest: R² = {r2_rf:.3f}, RMSE = {rmse_rf:,.0f}

2. OBSERVAÇÕES:
   - O modelo Random Forest {'apresentou melhor desempenho geral' if r2_rf > r2_lr else 'teve desempenho similar'}
   - O R² de {max(r2_lr, r2_rf):.3f} indica que o modelo explica aproximadamente 
     {max(r2_lr, r2_rf)*100:.1f}% da variância nos preços
   - O RMSE de {min(rmse_lr, rmse_rf):,.0f} representa o erro médio nas previsões

3. VARIÁVEIS MAIS IMPORTANTES (Random Forest):
   - {feature_importance_rf.iloc[0]['feature']}: {feature_importance_rf.iloc[0]['importance']:.1%}
   - {feature_importance_rf.iloc[1]['feature']}: {feature_importance_rf.iloc[1]['importance']:.1%}
   - {feature_importance_rf.iloc[2]['feature']}: {feature_importance_rf.iloc[2]['importance']:.1%}

4. ANÁLISE DE RESÍDUOS:
   - Os resíduos devem estar distribuídos aleatoriamente em torno de zero
   - Padrões nos resíduos podem indicar relações não capturadas pelos modelos

5. RECOMENDAÇÕES:
   - Utilizar o modelo Random Forest para previsões futuras
   - Considerar coletar mais dados ou features adicionais para melhorar a precisão
   - Validar o modelo com novos dados para verificar generalização

CONCLUSÃO:
O projeto demonstra a aplicação prática de técnicas de machine learning
para prever preços de imóveis usando dados reais, com ambos os modelos
mostrando resultados promissores para auxiliar na valuation de propriedades.
""")

# Exemplo de previsão para um novo imóvel
print("\n" + "="*50)
print("EXEMPLO DE PREVISÃO PARA UM NOVO IMÓVEL")
print("="*50)

# Criando um exemplo de novo imóvel com as features disponíveis
novo_imovel_data = {}
for feature in available_features:
    # Valores padrão baseados no tipo de feature
    if 'surface' in feature:
        novo_imovel_data[feature] = [120]
    elif 'price_usd_per_m2' in feature:
        novo_imovel_data[feature] = [2500]
    elif 'latitude' in feature:
        novo_imovel_data[feature] = [-34.58]
    elif 'longitude' in feature:
        novo_imovel_data[feature] = [-58.44]
    elif 'area_' in feature:
        novo_imovel_data[feature] = [10] if 'difference' in feature else [0.92]
    else:
        # Para features codificadas, usar valor médio
        novo_imovel_data[feature] = [1]

novo_imovel = pd.DataFrame(novo_imovel_data)

# Garantindo a mesma ordem das colunas
novo_imovel = novo_imovel[available_features]

previsao_lr = lr_model.predict(scaler.transform(novo_imovel))[0]
previsao_rf = rf_model.predict(novo_imovel)[0]

print(f"Características do imóvel:")
for feature in available_features[:5]:  # Mostrar apenas as 5 primeiras
    print(f"- {feature}: {novo_imovel[feature].iloc[0]}")

print(f"\nPrevisão de preço:")
print(f"- Regressão Linear: {previsao_lr:,.2f}")
print(f"- Random Forest: {previsao_rf:,.2f}")
print(f"- Previsão média: {(previsao_lr + previsao_rf)/2:,.2f}")

print("\n" + "="*70)
print("PROJETO CONCLUÍDO COM SUCESSO!")
print("="*70)
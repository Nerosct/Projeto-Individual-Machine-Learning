# Imagem base do Python
FROM python:3.10-slim

# Definir diretório de trabalho dentro do container
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o restante do projeto
COPY . .

# Comando padrão ao iniciar o container
CMD ["python", "src/Main.py"]

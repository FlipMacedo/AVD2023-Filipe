import spacy
from collections import Counter
from nltk.corpus import stopwords
import string
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import webbrowser
from pathlib import Path
import pandas as pd

# Carregar o modelo de língua portuguesa
nlp = spacy.load("pt_core_news_sm")

# Lista de stopwords em português
stop_words = set(stopwords.words("portuguese"))

# Definir o diretório de saída para os gráficos gerados e a página HTML
output_dir = Path("output")

def analyze_file(filename):
    text = read_file(filename)
    cleaned_text = clean_text(text)
    doc = nlp(cleaned_text)

    people = extract_people(doc)  # Extrair pessoas
    places = extract_places(doc)  # Extrair locais
    orgs = extract_orgs(doc)  # Extrair organizações
    lemmas = extract_lemmas(doc)  # Extrair lemas
    mwe = extract_mwe(doc)  # Extrair MWE (Multi-Word Expressions)
    keywords = extract_keywords(doc)  # Extrair palavras-chave
    dates = extract_dates(doc)  # Extrair datas
    sentiment = sentiment_analysis(cleaned_text)  # Realizar análise de sentimento

    # Imprimir os resultados da análise
    print(f"\n===== Análise de {filename} =====")
    print("Top 10 Pessoas:")
    for person, count in Counter(people).most_common(10):
        print(f"{person}: {count}")
    print("\nTop 10 Locais:")
    for place, count in Counter(places).most_common(10):
        print(f"{place}: {count}")
    print("\nTop 10 Organizações:")
    for org, count in Counter(orgs).most_common(10):
        print(f"{org}: {count}")
    print("\nTop 10 Lemas:")
    for lemma, count in Counter(lemmas).most_common(10):
        print(f"{lemma}: {count}")
    print("\nTop 10 MWE:")
    for mwe_phrase, count in Counter(mwe).most_common(10):
        print(f"{mwe_phrase}: {count}")
    print("\nTop 10 Palavras-chave:")
    for keyword, count in Counter(keywords).most_common(10):
        print(f"{keyword}: {count}")
    print("\nTop 10 Datas:")
    for date, count in Counter(dates).most_common(10):
        print(f"{date}: {count}")
    print("\nAnálise de Sentimento:")
    print(f"Sentimento geral: {sentiment}")

    # Exportar os resultados da análise para CSV e Excel
    export_to_csv(people, "Pessoas.csv")
    export_to_csv(places, "Locais.csv")
    export_to_csv(orgs, "Organizacoes.csv")
    export_to_csv(lemmas, "Lemas.csv")
    export_to_csv(mwe, "MWE.csv")
    export_to_csv(keywords, "PalavrasChave.csv")
    export_to_csv(dates, "Datas.csv")
    export_to_excel({
        "Pessoas": Counter(people).most_common(10),
        "Locais": Counter(places).most_common(10),
        "Organizações": Counter(orgs).most_common(10),
        "Lemas": Counter(lemmas).most_common(10),
        "MWE": Counter(mwe).most_common(10),
        "Palavras-chave": Counter(keywords).most_common(10),
        "Datas": Counter(dates).most_common(10)
    }, "ResultadosAnalise.xlsx")

    # Criar gráficos de barras para cada top 10
    create_bar_plot(Counter(people).most_common(10), f"Top 10 Pessoas em {filename}", f"{Path(filename).stem}_pessoas.png")
    create_bar_plot(Counter(places).most_common(10), f"Top 10 Locais em {filename}", f"{Path(filename).stem}_locais.png")
    create_bar_plot(Counter(orgs).most_common(10), f"Top 10 Organizações em {filename}", f"{Path(filename).stem}_organizacoes.png")
    create_bar_plot(Counter(lemmas).most_common(10), f"Top 10 Lemas em {filename}", f"{Path(filename).stem}_lemas.png")
    create_bar_plot(Counter(mwe).most_common(10), f"Top 10 MWE em {filename}", f"{Path(filename).stem}_mwe.png")
    create_bar_plot(Counter(keywords).most_common(10), f"Top 10 Palavras-chave em {filename}", f"{Path(filename).stem}_palavraschave.png")
    create_bar_plot(Counter(dates).most_common(10), f"Top 10 Datas em {filename}", f"{Path(filename).stem}_datas.png")

    # Gerar página HTML
    generate_html_page(sentiment)

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()
    # Remover caracteres e símbolos desnecessários
    text = re.sub(r"\n+", " ", text)  # Substituir quebras de linha por espaços
    text = re.sub(r"\s+", " ", text)  # Substituir múltiplos espaços por um único espaço
    return text


def clean_text(text):
    # Remover pontuação e converter para minúsculas
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remover pontuação
    text = text.lower()  # Converter para minúsculas
    return text


def extract_people(doc):
    people = []
    for entity in doc.ents:
        if entity.label_ == "PER":  # Se a entidade for uma pessoa
            people.append(entity.text)  # Adicionar o texto da entidade à lista
    return Counter(people)  # Retornar um objeto Counter com a contagem das pessoas


def extract_places(doc):
    places = []
    for entity in doc.ents:
        if entity.label_ == "LOC":  # Se a entidade for um local
            places.append(entity.text)  # Adicionar o texto da entidade à lista
    return Counter(places)  # Retornar um objeto Counter com a contagem dos locais


def extract_orgs(doc):
    orgs = []
    for entity in doc.ents:
        if entity.label_ == "ORG":  # Se a entidade for uma organização
            orgs.append(entity.text)  # Adicionar o texto da entidade à lista
    return Counter(orgs)  # Retornar um objeto Counter com a contagem das organizações


def extract_lemmas(doc):
    lemmas = []
    for token in doc:
        if token.is_alpha and not token.is_stop:  # Se o token for uma palavra e não for uma stop word
            lemmas.append(token.lemma_)  # Adicionar o lemma do token à lista
    return Counter(lemmas)  # Retornar um objeto Counter com a contagem dos lemas


def extract_mwe(doc):
    mwe = []
    for chunk in doc.noun_chunks:
        if len(chunk) > 1:  # Se o chunk tiver mais de uma palavra
            mwe.append(chunk.text)  # Adicionar o texto do chunk à lista
    return Counter(mwe)  # Retornar um objeto Counter com a contagem das MWE (Multi-Word Expressions)


def extract_keywords(doc):
    keywords = []
    for chunk in doc.noun_chunks:
        if not any(token.is_stop for token in chunk) and chunk.root.is_alpha:
            # Se nenhum dos tokens do chunk for uma stop word e a raiz do chunk for uma palavra
            keywords.append(chunk.text)  # Adicionar o texto do chunk à lista
    return Counter(keywords)  # Retornar um objeto Counter com a contagem das palavras-chave


def extract_dates(doc):
    dates = []
    for entity in doc.ents:
        if entity.label_ == "DATE":  # Se a entidade for uma data
            dates.append(entity.text)  # Adicionar o texto da entidade à lista
    return Counter(dates)  # Retornar um objeto Counter com a contagem das datas


def sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = sentiment_scores["compound"]
    if compound_score >= 0.05:
        sentiment = "positive"  # Sentimento positivo
    elif compound_score <= -0.05:
        sentiment = "negative"  # Sentimento negativo
    else:
        sentiment = "neutral"  # Sentimento neutro
    return sentiment

def create_bar_plot(data, title, filename):
    categories = [item[0] for item in data]  # Obter as categorias do gráfico a partir dos dados fornecidos
    counts = [item[1] for item in data]  # Obter as contagens do gráfico a partir dos dados fornecidos

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(categories)), counts, align="center")  # Criar um gráfico de barras horizontais
    plt.yticks(range(len(categories)), categories)  # Definir os rótulos do eixo y como as categorias
    plt.xlabel("Frequência")  # Definir o rótulo do eixo x como "Frequência"
    plt.title(title)  # Definir o título do gráfico
    plt.tight_layout()
    plt.savefig(filename)  
    plt.close()


def generate_html_page(sentiment):
    html_content = """
        <html>
        <head>
            <title>Resultados da Análise</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                }
                h1 {
                    text-align: center;
                    margin-bottom: 30px;
                }
                h2 {
                    margin-top: 50px;
                    margin-bottom: 10px;
                }
                img {
                    display: block;
                    margin-left: auto;
                    margin-right: auto;
                    margin-top: 20px;
                    max-width: 80%;
                    height: auto;
                }
            </style>
        </head>
        <body>
        """
    html_content += f"<h1>Sentimento Geral: {sentiment}</h1>"  # Include the overall sentiment in the title

    image_files = sorted(output_dir.glob("*_*.png"))  # Sort the image files by name
    for image_file in image_files:
        plot_title = image_file.stem.replace("_", " ")  # Get the plot title from the file name
        image_path = image_file.relative_to(output_dir)  # Get the relative path of the image relative to the output directory
        html_content += f"<h2>{plot_title}</h2>"
        html_content += f'<img src="{image_path}" alt="{plot_title}"><br><br>'

    html_content += "</body></html>"

    with open(output_dir / "resultados_analise.html", "w", encoding="utf-8") as file:
        file.write(html_content)

    webbrowser.open_new_tab(output_dir / "resultados_analise.html")  # Open the generated HTML page in the default web browser
 # Abrir a página HTML gerada no Browser


def export_to_csv(data, filename):
    data_with_columns = [(item, count) for item, count in data.items()]
    df = pd.DataFrame(data_with_columns, columns=["Item", "Contagem"])  # Criar um DataFrame com as colunas "Item" e "Contagem"
    df.to_csv(output_dir / filename, index=False)  # Exportar o DataFrame para CSV


def export_to_excel(data_dict, filename):
    with pd.ExcelWriter(output_dir / filename) as writer:
        for sheet_name, data in data_dict.items():
            df = pd.DataFrame(data, columns=["Item", "Contagem"])  # Criar um DataFrame com as colunas "Item" e "Contagem"


            df.to_excel(writer, sheet_name=sheet_name, index=False)  # Exportar o DataFrame para Excel


analyze_file("C:/Filipe/MHD_Laptop/Analise e Viz. Dados/FinalP/Camilo/Camilo/Obra/Camilo-A_mulher_fatal.txt")

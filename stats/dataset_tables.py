import argparse
from pathlib import Path

import pandas as pd

STATS_DIR = Path(__file__).resolve().parent
CSV_PATH = STATS_DIR.parent / 'Spanish_ClickbaitCorpus_2026.csv'
OUTPUT_DIR = STATS_DIR / 'dataset_tables'
SOURCE = 'Newspaper'
REGION = 'Region'
REGIONS = ['Spain', 'Latin America']
SPLITS = {'Train': 'train', 'Validation': 'val', 'Test': 'test'}
TRANSLATIONS = {
    'Actualidad': 'Current affairs', 'Automoción': 'Automotive',
    'Belleza': 'Beauty', 'Ciencia': 'Science', 'Cine': 'Cinema',
    'Clima': 'Weather', 'Deportes': 'Sports', 'Economía': 'Economy',
    'Gastronomía': 'Gastronomy', 'Legislación': 'Legislation',
    'Lenguaje': 'Language', 'Música': 'Music', 'Política': 'Politics',
    'Salud': 'Health', 'Seguridad': 'Security', 'Tecnología': 'Technology',
    'Televisión': 'Television', 'Turismo': 'Tourism',
    'Videojuegos': 'Video games',
}


def load_data(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    required = {'Class', SOURCE, REGION, 'Title', 'Body', 'Response', '_split'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    for col in ['Class', SOURCE, REGION]:
        df[col] = df[col].astype('string').str.strip()
    if df[['Class', SOURCE, REGION]].isna().any().any():
        raise ValueError('Missing category, source, or region values.')
    if not df[REGION].isin(REGIONS).all():
        raise ValueError('Invalid region values.')
    if (df.groupby(SOURCE)[REGION].nunique() > 1).any():
        raise ValueError('Some sources are assigned to multiple regions.')
    df['Category'] = df['Class'].replace(TRANSLATIONS)
    return df


def write_tex(output_dir, filename, lines):
    (output_dir / filename).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def word_counts(series):
    return series.fillna('').astype(str).str.split().str.len()


def dataset_statistics(df, output_dir):
    stats = {}
    for name, value in SPLITS.items():
        subset = df[df['_split'] == value]
        if subset.empty:
            raise ValueError(f'Empty split: {value}')
        headline = word_counts(subset['Title'])
        article = word_counts(subset['Body'])
        answer = word_counts(subset['Response'])
        source_counts = subset[SOURCE].value_counts()
        stats[name] = {
            'Number of articles': len(subset),
            'Headline mean': round(headline.mean()),
            'Article mean': round(article.mean()),
            'Article median': round(article.median()),
            'Article maximum': int(article.max()),
            'Answer mean': round(answer.mean()),
            'Answer median': round(answer.median()),
            'Answer P90': round(answer.quantile(0.90)),
            'Answer maximum': int(answer.max()),
            'Unique sources': subset[SOURCE].nunique(),
            'Largest source share': source_counts.iloc[0] / len(subset) * 100,
        }

    def row(label, key, decimals=0):
        values = [f'{stats[name][key]:.{decimals}f}' if decimals else str(int(stats[name][key])) for name in SPLITS]
        return f'{label} & ' + ' & '.join(values) + r' \\'

    lines = [
        r'\begin{table}[ht]', r'\centering', r'\begin{tabular}{lrrr}',
        r'\toprule', r' & \textbf{Train} & \textbf{Validation} & \textbf{Test} \\',
        r'\midrule', row(r'\textbf{Number of articles}', 'Number of articles'),
        r'\midrule', r'\multicolumn{4}{l}{\textbf{Headline length (words)}} \\',
        row('Mean', 'Headline mean'), r'\midrule',
        r'\multicolumn{4}{l}{\textbf{Article length (words)}} \\',
        row('Mean', 'Article mean'), row('Median', 'Article median'), row('Maximum', 'Article maximum'),
        r'\midrule', r'\multicolumn{4}{l}{\textbf{Answer length (words)}} \\',
        row('Mean', 'Answer mean'), row('Median', 'Answer median'),
        row('90th percentile', 'Answer P90'), row('Maximum', 'Answer maximum'),
        r'\midrule', r'\multicolumn{4}{l}{\textbf{Source distribution}} \\',
        row('Number of unique sources', 'Unique sources'),
        row(r'Largest source share (\%)', 'Largest source share', decimals=2),
        r'\bottomrule', r'\end{tabular}',
        r'\caption{Dataset statistics for the training, validation and test splits of Spanish ClickbaitCorpus 2026. Text lengths are measured in words. The largest source share represents the percentage of articles contributed by the most frequent news outlet in each split.}',
        r'\label{tab:wordcounts}', r'\end{table}',
    ]
    write_tex(output_dir, 'dataset_statistics.tex', lines)


def geographical_distribution(df, output_dir):
    sources = df[[SOURCE, REGION]].drop_duplicates(subset=[SOURCE])
    source_counts = sources[REGION].value_counts().reindex(REGIONS, fill_value=0)
    article_counts = df[REGION].value_counts().reindex(REGIONS, fill_value=0)
    lines = [
        r'\begin{table}[ht]', r'\centering', r'\begin{tabular}{lrrrr}',
        r'\toprule',
        r'\textbf{Region} & \textbf{Sources} & \textbf{Sources (\%)} & \textbf{Articles} & \textbf{Articles (\%)} \\',
        r'\midrule',
    ]
    for region in REGIONS:
        s, a = int(source_counts[region]), int(article_counts[region])
        lines.append(f'{region} & {s} & {s / len(sources) * 100:.2f} & {a} & {a / len(df) * 100:.2f}' + r' \\')
    lines += [
        r'\midrule',
        rf'\textbf{{Total}} & \textbf{{{len(sources)}}} & \textbf{{100.00}} & \textbf{{{len(df):,}}} & \textbf{{100.00}} \\',
        r'\bottomrule', r'\end{tabular}',
        r'\caption{Geographical distribution of news outlets and articles in Spanish ClickbaitCorpus 2026. Source percentages are calculated over unique news outlets, while article percentages are calculated over all articles.}',
        r'\label{tab:geographical-distribution}', r'\end{table}',
    ]
    write_tex(output_dir, 'geographical_distribution.tex', lines)


def category_region_distribution(df, output_dir):
    counts = df.groupby(['Category', REGION]).size().unstack(fill_value=0).reindex(columns=REGIONS, fill_value=0)
    counts['Total'] = counts.sum(axis=1)
    counts = counts.sort_values('Total', ascending=False)
    lines = [
        r'\begin{table}[ht]', r'\centering', r'\small', r'\begin{tabular}{lrrr}',
        r'\toprule',
        r'\textbf{Category} & \textbf{Spain} & \textbf{Latin America} & \textbf{Total} \\',
        r'\midrule',
    ]
    for category, row in counts.iterrows():
        spain, latam, total = int(row['Spain']), int(row['Latin America']), int(row['Total'])
        spain_cell = f'{spain} ({spain / total * 100:.2f}\\%)'
        latam_cell = f'{latam} ({latam / total * 100:.2f}\\%)'
        lines.append(f'{category} & {spain_cell} & {latam_cell} & {total}' + r' \\')

    total_spain = int(counts['Spain'].sum())
    total_latam = int(counts['Latin America'].sum())
    total_articles = total_spain + total_latam
    lines += [
        r'\midrule',
        (rf'\textbf{{Total}} & \textbf{{{total_spain} ({total_spain / total_articles * 100:.2f}\%)}}'
         rf' & \textbf{{{total_latam} ({total_latam / total_articles * 100:.2f}\%)}}'
         rf' & \textbf{{{total_articles:,}}} \\'),
        r'\bottomrule', r'\end{tabular}',
        r'\caption{Distribution of articles across news categories and geographical groups in Spanish ClickbaitCorpus 2026. Each regional column reports the number of articles and, in parentheses, the percentage of articles within the corresponding category. Percentages in the total row are calculated over the complete corpus.}',
        r'\label{tab:category-region-distribution}', r'\end{table}',
    ]
    write_tex(output_dir, 'category_region_distribution.tex', lines)


def main():
    parser = argparse.ArgumentParser(description='Generate dataset tables.')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=OUTPUT_DIR,
        help=f'Output directory for LaTeX tables (default: {OUTPUT_DIR}).',
    )
    parser.add_argument(
        '--csv-path',
        type=Path,
        default=CSV_PATH,
        help=f'Input dataset CSV (default: {CSV_PATH}).',
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(args.csv_path)
    dataset_statistics(df, args.output_dir)
    geographical_distribution(df, args.output_dir)
    category_region_distribution(df, args.output_dir)


if __name__ == '__main__':
    main()

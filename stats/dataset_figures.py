import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

STATS_DIR = Path(__file__).resolve().parent
CSV_PATH = STATS_DIR.parent / 'Spanish_ClickbaitCorpus_2026.csv'
OUTPUT_DIR = STATS_DIR / 'dataset_figures'
TOP_N = 19
SOURCE_COLUMN = 'Newspaper'
REGION_COLUMN = 'Region'
CATEGORY_COLUMN = 'Class'
REGIONS = ['Spain', 'Latin America']
REGION_COLORS = {'Spain': '#4472C4', 'Latin America': '#EDC948', 'Others': '#BAB0AC'}
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

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def load_data(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    required = {SOURCE_COLUMN, REGION_COLUMN, CATEGORY_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    for col in required:
        df[col] = df[col].astype('string').str.strip()
    if df[list(required)].isna().any().any():
        raise ValueError('Missing category, source, or region values.')
    if not df[REGION_COLUMN].isin(REGIONS).all():
        raise ValueError('Invalid region values.')
    if (df.groupby(SOURCE_COLUMN)[REGION_COLUMN].nunique() > 1).any():
        raise ValueError('Some sources are assigned to multiple regions.')
    df['Category'] = df[CATEGORY_COLUMN].replace(TRANSLATIONS)
    return df


def plot_categories(df, output_dir):
    distribution = df.groupby(['Category', REGION_COLUMN]).size().unstack(fill_value=0)
    distribution = distribution.reindex(columns=REGIONS, fill_value=0)
    distribution['Total'] = distribution.sum(axis=1)
    distribution = distribution.sort_values('Total', ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.barh(distribution.index, distribution['Spain'], color=REGION_COLORS['Spain'], height=0.7, label='Spain')
    ax.barh(distribution.index, distribution['Latin America'], left=distribution['Spain'], color=REGION_COLORS['Latin America'], height=0.7, label='Latin America')
    ax.invert_yaxis()
    for i, total in enumerate(distribution['Total']):
        ax.text(total + 4, i, f'{total} ({total / len(df) * 100:.2f}%)', va='center', ha='left', fontsize=9)
    ax.set_xlabel('Number of articles')
    ax.set_xlim(0, distribution['Total'].max() * 1.30)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / 'Fig1.pdf', bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)


def plot_regions(df, output_dir):
    sources = df[[SOURCE_COLUMN, REGION_COLUMN]].drop_duplicates(subset=[SOURCE_COLUMN])
    counts = sources[REGION_COLUMN].value_counts().reindex(REGIONS, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    _, _, autotexts = ax.pie(
        counts, labels=REGIONS, colors=[REGION_COLORS[r] for r in REGIONS],
        autopct=lambda pct: f'{pct:.1f}%\n({round(pct * len(sources) / 100)} sources)',
        startangle=90, counterclock=False,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
        pctdistance=0.65, labeldistance=1.08, textprops={'fontsize': 11},
    )
    for t in autotexts:
        t.set_fontweight('bold')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(output_dir / 'Fig2.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_sources(df, output_dir):
    sources = df[[SOURCE_COLUMN, REGION_COLUMN]].drop_duplicates(subset=[SOURCE_COLUMN])
    counts = df[SOURCE_COLUMN].value_counts().rename_axis('Source').reset_index(name='Articles')
    counts = counts.merge(sources.rename(columns={SOURCE_COLUMN: 'Source'}), on='Source', validate='one_to_one')
    counts['Percentage'] = counts['Articles'] / len(df) * 100
    n = min(TOP_N, len(counts))
    top = counts.head(n)
    others_count = int(counts.iloc[n:]['Articles'].sum())
    if len(counts) > n:
        others = pd.DataFrame([{'Source': 'Others', 'Articles': others_count, REGION_COLUMN: 'Others', 'Percentage': others_count / len(df) * 100}])
        data = pd.concat([top, others], ignore_index=True)
        positions = list(range(n)) + [n + 1]
    else:
        data = top
        positions = list(range(n))
    fig, ax = plt.subplots(figsize=(9, 8))
    bars = ax.barh(positions, data['Articles'], color=[REGION_COLORS[r] for r in data[REGION_COLUMN]], height=0.65)
    ax.set_yticks(positions)
    ax.set_yticklabels(data['Source'])
    ax.invert_yaxis()
    for bar, count, pct in zip(bars, data['Articles'], data['Percentage']):
        ax.text(bar.get_width() + 4, bar.get_y() + bar.get_height() / 2, f'{count} ({pct:.2f}%)', va='center', ha='left', fontsize=9)
    ax.set_xlabel('Number of articles')
    ax.set_xlim(0, data['Articles'].max() * 1.22)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.2)
    ax.set_axisbelow(True)
    if len(counts) > n:
        ax.axhline(y=n - 0.05, color='#BBBBBB', linewidth=0.8, linestyle='--')
    legend = [Patch(facecolor=REGION_COLORS[r], label=label) for r, label in [('Spain', 'Spain'), ('Latin America', 'Latin America'), ('Others', 'Other sources (aggregated)')]]
    ax.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / 'Fig3.pdf', bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Generate dataset figures.')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=OUTPUT_DIR,
        help=f'Output directory for PDF figures (default: {OUTPUT_DIR}).',
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
    plot_categories(df, args.output_dir)
    plot_regions(df, args.output_dir)
    plot_sources(df, args.output_dir)


if __name__ == '__main__':
    main()

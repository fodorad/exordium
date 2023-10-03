from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def df_to_image(df: pd.DataFrame,
                output_path: str | Path,
                ax: plt.Axes | None = None,
                column_width: float = 2.0,
                row_height: float = 0.625,
                font_size: int = 14,
                header_color: str = '#110a1f',
                row_colors: list[str] = ['#f1f1f2', 'w'],
                edge_color: str = 'w',
                bbox: list = [0, 0, 1, 1],
                header_columns: int = 0,
                verbose: bool = True) -> None:

        if ax is None:
            size = (np.array(df.shape[::-1]) + np.array([0, 1])) * np.array([column_width, row_height])
            _, ax = plt.subplots(figsize=size)
            ax.axis('off')

        table = ax.table(cellText=df.values, bbox=bbox, colLabels=df.columns)
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)

        for k, cell in table._cells.items():
            cell.set_edgecolor(edge_color)
            if k[0] == 0 or k[1] < header_columns:
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(row_colors[k[0]%len(row_colors)])

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path))
        plt.close()

        if verbose:
            print(f'Table is saved to: {output_path}')


if __name__ == '__main__':
    d = {
        'Method': ['SVM', 'CNN'],
        'P': ['x', 'y'],
        'R': [0.92, 0.95],
        'F1': [0.93, 0.97],
    }
    df = pd.DataFrame(data=d)
    df_to_image(df, 'test.png')
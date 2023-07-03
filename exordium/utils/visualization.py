from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from exordium.utils.shared import reload_matplotlib



def create_annotation(data: np.ndarray,
                      length: int,
                      height: int,
                      width: int,
                      target_name: str | None = None,
                      win_size: int = 61,
                      fps: int = 30,
                      interval = (0, 1)):

    assert data.ndim == 2 and data.shape[-1] == 1

    # white background
    bg = (np.ones(shape=(length, height, width, 3))*255).astype(np.uint8)

    # grey middle lines
    bg[:,bg.shape[1]//2,:] = (200,200,200)
    bg[:,bg.shape[1]//4,:] = (230,230,230)
    bg[:,bg.shape[1]//4*3,:] = (230,230,230)

    # black border
    bg[:,0,:,:] =  0
    bg[:,-1,:,:] = 0
    bg[:,:,0,:] =  0
    bg[:,:,-1,:] = 0

    # red current timestamp
    bg[:,:,bg.shape[2]//2] = (255,0,0)
    dict_putText = {'fontFace': cv2.FONT_HERSHEY_SIMPLEX, 'fontScale': 0.4, 'color': (255,0,0), 'thickness': 1, 'lineType': cv2.LINE_4}

    # create annotation dot at position x,y
    def add_dot(t, x, y, size: int = 3):
        border = size // 2
        x_min = x-border
        if x_min < 0: x_min = 0
        x_max = x+border+1
        if x_max > bg.shape[2]: x_max = bg.shape[2]
        y_min = y-border
        if y_min < 0: y_min = 0
        y_max = y+border+1
        if y_max > bg.shape[1]: y_max = bg.shape[1]
        bg[t, y_min:y_max, x_min:x_max, :] = 0

    # windowing annotation at timestamp t
    def create_annotation_window(timestamp: int):
        ann_t = np.empty(shape=(win_size,))
        ann_t[:] = np.nan
        start = timestamp - win_size // 2
        prepad = False

        if start < 0: 
            start = 0
            prepad = True

        end = timestamp + win_size // 2 + 1
        postpad = False

        if end > data.shape[0]: 
            end = data.shape[0] # data : 1 value / sec
            postpad = True

        w = data[start:end,0]
        if postpad or (not prepad and not postpad):
            ann_t[:w.shape[0]] = w
        else:
            ann_t[-w.shape[0]:] = w

        return ann_t


    # save on figure
    def plot_annotation_window(ann_t, t):
        assert ann_t.ndim == 1
        plot_x = np.linspace(0, width, win_size).astype(np.int32)
        # width == 64, win_size == 11: 
        # plot_x == [  0.,  64., 128., 192., 256., 320., 384., 448., 512., 576., 640.]

        for ind, x_val in enumerate(list(plot_x)):
            if not np.isnan(ann_t[ind]):
                true_val = ann_t[ind]
                y_val = height - int(height * np.clip(true_val, 0, 1)) # inverted because of cv2
                current_val = np.round(ann_t[len(ann_t)//2], decimals=2)

                for t_val in range(t*fps, t*fps+fps):
                    add_dot(t_val, x_val, y_val)

                    if not np.isnan(ann_t[len(ann_t)//2]):
                        bg[t_val,...] = cv2.putText(bg[t_val,...], f'{current_val}', (width//2+5,10), **dict_putText)


    for t in range(length//fps):
        ann_t = create_annotation_window(t)
        #print(t, ann_t)
        plot_annotation_window(ann_t, t)

    if target_name is not None:
        for t in range(length):
            bg[t,...] = cv2.putText(bg[t,...], f'{target_name}', (5,10), **dict_putText)

    return bg


def add_to_videos(video, annotation, position: tuple = (5, 5)):
    '''Add annotation to video

    Arguments:
        video (np.ndarray): video represented as numpy array of shape (L, H, W, 3)
        annotation (np.ndarray): annotation represented as numpy array of shape (L, H, W, 3)
        position (tuple): top left x, y coordinates of the annotation window  
    '''
    video[:,position[0]:position[0]+annotation.shape[1],position[1]:position[1]+annotation.shape[1],:] = annotation


def _addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')


def visualize_hists(data: list[dict], x_min: int = None, x_max: int = None, x_step: int = 5, title: str = 'Hists', output_path: str = 'test.png') -> None:
    yaw = np.array([elem['headpose'][0] for elem in data])
    pitch = np.array([elem['headpose'][1] for elem in data])
    roll = np.array([elem['headpose'][2] for elem in data])

    if x_min is None:
        x_min_tmp = np.nanmin(np.concatenate([yaw, pitch, roll]))
        x_min = np.around(x_min_tmp, decimals=-1)
        if x_min > x_min_tmp:
            x_min -= 10

    if x_max is None:
        x_max_tmp = np.nanmax(np.concatenate([yaw, pitch, roll]))
        x_max = np.around(x_max_tmp, decimals=-1)
        if x_max < x_max_tmp:
            x_max += 10

    for name, d in zip(['Yaw', 'Pitch', 'Roll'], [yaw, pitch, roll]):
        h, b = np.histogram(d, bins=list(range(int(x_min), int(x_max+1), int(x_step))))
        names = [f"[{b[i]},{b[i+1]})" for i in range(len(b)-1)]
        fig = plt.figure(figsize=(10,5))
        plt.bar(names, h)
        _addlabels(names, h)
        plt.title(f'{title} {name}')
        fig.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(str(output_path.parent / f'{output_path.stem}_{name}.png'))
        plt.close()


def visualize_headposes(data: list[dict], title: str = 'Headposes: Yaw, Pitch, Roll', output_path: str | Path = 'headposes.png') -> None:
    assert isinstance(data, list) and len(data) > 0, \
        f'Invalid input format. Expected: list of dicts, got instead {type(data)}'
    assert isinstance(data[0], dict) and 'frame_path' in data[0] and 'headpose' in data[0], \
        f'Invalid input element format. Expected: dicts with "frame_path" and "headpose" as keys, got instead {type(data)}, elems: {data[0]}'

    yaw = np.array([elem['headpose'][0] for elem in data])
    pitch = np.array([elem['headpose'][1] for elem in data])
    roll = np.array([elem['headpose'][2] for elem in data])

    elems_yaw = [data[np.nanargmin(np.abs(yaw - cp))] for cp in np.linspace(np.nanmin(yaw), np.nanmax(yaw), num=10)]
    elems_pitch = [data[np.nanargmin(np.abs(pitch - cp))] for cp in np.linspace(np.nanmin(pitch), np.nanmax(pitch), num=10)]
    elems_roll = [data[np.nanargmin(np.abs(roll - cp))] for cp in np.linspace(np.nanmin(roll), np.nanmax(roll), num=10)]

    fig, axs = plt.subplots(3, 10, figsize=(15, 5))
    for i in range(10):
        axs[0, i].imshow(cv2.cvtColor(cv2.imread(elems_yaw[i]['frame_path']), cv2.COLOR_BGR2RGB))
        axs[0, i].set_title(int(elems_yaw[i]['headpose'][0]))
        axs[1, i].imshow(cv2.cvtColor(cv2.imread(elems_pitch[i]['frame_path']), cv2.COLOR_BGR2RGB))
        axs[1, i].set_title(int(elems_pitch[i]['headpose'][1]))
        axs[2, i].imshow(cv2.cvtColor(cv2.imread(elems_roll[i]['frame_path']), cv2.COLOR_BGR2RGB))
        axs[2, i].set_title(int(elems_roll[i]['headpose'][2]))

    for ax in axs.ravel():
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle(title)
    plt.savefig(str(output_path))
    plt.close()


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

        reload_matplotlib() # this fixes the headless mpl error that may occour during a lot of image exporting

        if ax is None:
            size = (np.array(df.shape[::-1]) + np.array([0, 1])) * np.array([column_width, row_height])
            fig, ax = plt.subplots(figsize=size)
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
    pass
    #from exordium.video.frames import *
    #add_to_videos(new_video, line, p_ind, t_ind)
    #save_video(new_video, audio, video_output_path)
    #print(f'G{group_id} saved: {video_output_path}')
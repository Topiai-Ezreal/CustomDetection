import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def selective_search(img, strategy):
    """
    选择性搜索
    single、fast、quality分别为论文中的三个搜索模式
    """
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)

    if strategy == 'single':
        ss.switchToSingleStrategy()
    elif strategy == 'fast':
        ss.switchToSelectiveSearchFast()
    elif strategy == 'quality':
        ss.switchToSelectiveSearchQuality()
    else:
        raise TypeError('strategy not exist')

    rects = ss.process()
    rects[:, 2] += rects[:, 0]
    rects[:, 3] += rects[:, 1]
    return rects


def visualize_proposals(img, rects):
    """
    可视化proposals
    """
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in rects:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()


# img = cv2.imdecode(np.fromfile('E:\\1_database\\APTOS_tianchi\\phase2\\train\\2L_Post_10.jpg', dtype=np.uint8), 1)
# rects = selective_search(img, 'fast')
# visualize_proposals(img, rects)

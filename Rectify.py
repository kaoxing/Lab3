import cv2
import numpy as np
from scipy.spatial import distance as dist


def remove_marginal_line(line, threshold, shape):
    """
    移除靠近图像边缘的直线
    """
    for i in range(len(line)):
        if np.abs(line[i][1] - shape[0]) < threshold or np.abs(line[i][0] - shape[1]) < threshold:
            return False
        if np.abs(line[i][1]) < threshold or np.abs(line[i][0]) < threshold:
            return False
    return True


def getLinearEquation(p1x, p1y, p2x, p2y):
    """计算直线解析方程 ax+by+c=0"""
    sign = 1
    a = p2y - p1y
    if a < 0:
        sign = -1
        a = sign * a
    b = sign * (p1x - p2x)
    c = sign * (p1y * p2x - p1x * p2y)
    return a, b, c


def elongate_line(line, shape):
    """
    延长直线直到图像边缘
    """
    a, b, c = getLinearEquation(line[0][0], line[0][1], line[1][0], line[1][1])
    # 计算直线到边缘的点
    y1 = -(a * 0 + c) // b
    y2 = -(a * shape[1] + c) // b
    return [(0, y1), (shape[1], y2)]


def get_cross_point(line1, line2, threshold):
    """
    当两条直线斜率之比大-1于threshold时，计算两条直线的交点，否则返回None
    这样是为了抑制同一条边上提取出的不同直线相交产生的点，这些点会落在边的中间，有可能会产生一系列连续点
    """
    printtttttttt()
    k1 = float(line1[0][1] - line1[1][1]) / float(line1[0][0] - line1[1][0])
    k2 = float(line2[0][1] - line2[1][1]) / float(line2[0][0] - line2[1][0])
    print("K", k1, k2)
    if np.abs((np.abs(k1) / np.abs(k2)) - 1) < threshold or k1 == k2:
        return None
    print("Not None")
    x1 = line1[0][0]  # 取四点坐标
    y1 = line1[0][1]
    x2 = line1[1][0]
    y2 = line1[1][1]

    x3 = line2[0][0]
    y3 = line2[0][1]
    x4 = line2[1][0]
    y4 = line2[1][1]

    k1 = (y2 - y1) * 1.0 / (x2 - x1)
    b1 = y1 * 1.0 - x1 * k1 * 1.0
    k2 = (y4 - y3) * 1.0 / (x4 - x3)
    b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 is None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return x, y


def remove_outer_points(points, shape):
    """
    移除图像外的点
    """
    ret = []
    for point in points:

        if point[0] > shape[1] or point[0] < 0:
            continue
        if point[1] > shape[0] or point[1] < 0:
            continue
        ret.append(point)
    return ret


def get_great_points(points, shape, block, threshold):
    """
    这里对图像大小的值全为1的像素矩阵进行投票，每个点四周的block个范围内的所有值*2
    然后进行全局抑制，对所有点的值-threshold，然后返回大于0的所有点的坐标
    """
    ret = []
    vote = np.ones(shape=shape, dtype=int)
    print(vote.shape)
    # print("before", points)
    for point in points:
        x = int(point[0])
        y = int(point[1])
        for i in range(block):
            for j in range(block):
                vote[x + i - block // 2][y + j - block // 2] += 1
        # vote[x][y] += 1
    vote = vote - threshold
    x, y = np.where(vote > 0)
    for i in range(len(x)):
        ret.append((x[i], y[i]))
    # print("after", ret)
    return ret


def remove_close_points(points, threshold):
    """
    抑制距离小于threshold的点
    """

    def cal_distance(point_1, point_2):
        # print(point_1)
        x1 = point_1[0]
        y1 = point_1[1]
        x2 = point_2[0]
        y2 = point_2[1]
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    newPoints = []
    for point1 in points:
        flag = True
        for point2 in newPoints:
            if cal_distance(point1, point2) < threshold:
                flag = False
                break
        if flag:
            newPoints.append([point1[0], point1[1]])
    return newPoints


def order_points(pts):
    # 根据点的 x 坐标对点进行排序
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # 从根据点的 x 坐标排序的坐标点中获取最左和最右的点
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # 现在，根据y坐标对最左边的坐标排序，这样我们就可以分别获取左上角和左下角的点
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # 现在我们有了左上角的坐标，用它作为锚点来计算左上角和右下角点之间的欧氏距离;根据勾股定理，距离最大的点就是右下点
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    # 按左上、右上、右下和左下顺序返回坐标
    return np.array([tl, tr, br, bl], dtype="float32")


def rectify(path,width,heigth):
    im0 = cv2.imread(path, cv2.IMREAD_COLOR)
    if im0 is None:
        print('read image failed')
        exit()

    im1 = im0[:, :, 0]
    _, im2 = cv2.threshold(im1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.namedWindow('threshold', cv2.WINDOW_NORMAL)
    cv2.imshow('threshold', im2)

    edges = cv2.Canny(im2, 50, 200, L2gradient=True)
    cv2.namedWindow('edge', cv2.WINDOW_NORMAL)
    cv2.imshow('edge', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    im = im0.copy()
    im1 = im0.copy()
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 130, minLineLength=100, maxLineGap=8)
    lines_temp = []
    N = lines.shape[0]
    for i in range(N):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]
        cv2.line(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if remove_marginal_line(((x1, y1), (x2, y2)), 50, im.shape):
            # 只保留距离图像边缘50像素以上的线
            lines_temp.append([(x1, y1), (x2, y2)])
            cv2.line(im1, (x1, y1), (x2, y2), (0, 255, 0), 2)
    lines = lines_temp.copy()  # 保留下来的线
    lines_temp.clear()

    cv2.namedWindow('raw', cv2.WINDOW_NORMAL)
    cv2.imshow('raw', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    im = im0.copy()

    cv2.namedWindow('delete_margin', cv2.WINDOW_NORMAL)
    cv2.imshow('delete_margin', im1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # for line in lines:
    #     temp = elongate_line(line, im.shape)
    #     lines_temp.append(elongate_line(line, im.shape))  # 延长每条线直到图像边缘
    #     cv2.line(im, temp[0], temp[1], (0, 255, 0), 2)
    # lines = lines_temp.copy()
    # lines_temp.clear()

    print(lines)

    points_cross = []  # 计算交点
    for line1 in lines:
        for line2 in lines:
            if line1 == line2:
                continue
            point = get_cross_point(line1, line2, 0.8)
            if point is None:
                continue
            points_cross.append(point)
            cv2.circle(im, (int(point[0]), int(point[1])), 7, (0, 0, 255), 4)

    cv2.namedWindow('before', cv2.WINDOW_NORMAL)
    cv2.imshow('before', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    im = im0.copy()
    points_cross = remove_outer_points(points_cross, im.shape)  # 移除图像外的点
    for point in points_cross:
        cv2.circle(im, (int(point[0]), int(point[1])), 7, (0, 0, 255), 1)
    cv2.namedWindow('before', cv2.WINDOW_NORMAL)
    cv2.imshow('before', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    im = im0.copy()

    # 这里用区间投票的方式，对以每个点为中心，三个像素内的点值*2，然后进行全局抑制，对所有点的值-1，然后返回大于0的所有点的坐标
    # 目的与之前的求交阈值相同，都是为了去除零散点
    pts_img = get_great_points(points_cross, (im.shape[1], im.shape[0]), 10, 1)
    pts_img = remove_close_points(pts_img, 200)  # 距离在300个像素内的点仅保留1个

    # pts_img = remove_close_points(points_cross, 200)


    print("pts", pts_img)
    pts_img = np.array(pts_img, dtype=int)

    # pts_img = np.array([(45, 118), (766, 79), (766, 79), (766, 79)], dtype=np.float32) - 1
    pts_obj = np.array([[0, 0], [width, 0], [width, heigth], [0, heigth]], dtype=np.float32) * 50

    obj_width, obj_height = pts_obj[2, 0], pts_obj[2, 1]

    for j in range(len(pts_img)):
        cv2.circle(im, (pts_img[j, 0], pts_img[j, 1]), 7, (0, 0, 255), 4)

    cv2.namedWindow('after', cv2.WINDOW_NORMAL)
    cv2.imshow('after', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    im = im0.copy()

    # pts_img = remove_close_points(pts_img, 300)  # 距离在300个像素内的点仅保留1个
    pts_img = np.array(pts_img, dtype=np.float32)
    pts_img = order_points(pts_img)

    print(pts_img)
    M = cv2.getPerspectiveTransform(pts_img, pts_obj)
    warped = cv2.warpPerspective(im0, M, (obj_width, obj_height))
    # print(warped.shape)
    cv2.namedWindow('warped', cv2.WINDOW_NORMAL)
    cv2.imshow('warped', warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    rectify("Lab3-3.jpg",21,26)

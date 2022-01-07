
# from numpy.core.fromnumeric import size
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import MobileUnet
import cv2
from post_process_tool import *
from math import ceil, pi
# model = MobileUnet.MobileUnet(1)
# torch.load('best.pt', map_location='cpu')
model = torch.load('cpu_diaphragm.pth')
model.eval()  # !如果未使用评估模式,BN层和dropout会有影响
depth = [860, 704, 848, 770, 770, 815,  735, 770]
# w_start = [0, 0, 0, 233, 0, 0,  285, 420]
# w_end = [1680, 1680, 1680, 1630, 1340, 1630,  1630, 1237]

# 不限制宽度区域的
w_start = [0, 0, 0, 0, 0, 0,  0, 0]
w_end = [1680, 1680, 1680, 1680, 1680, 1680,  1680, 1680]

half_h = 555
half_roi_height = 96

line_data = np.zeros((54, 1, 192))
line_label = np.zeros((54, 2, 192))
number = 0
for picture_num in range(1, 8):
    print(f'now it\'s picture {picture_num}')
    ScreenShot = Image.open(
        f"./dataset/raw_image/diaphragm ({picture_num}).jpg")
    # width, height = ScreenShot.size
    # (left, top, right, bottom) = (0, height*0.5, 1640, 1090)
    # croped_image = ScreenShot.crop((left, top, right, bottom))

    # roi_depth = find_roi_depth(croped_image)
    roi_depth = depth[picture_num]

    # width, height = croped_image.size
    (left, top, right, bottom) = (w_start[picture_num], roi_depth -
                                  half_roi_height, w_end[picture_num], roi_depth+half_roi_height)
    roi_area = ScreenShot.crop((left, top, right, bottom))
    width, height = roi_area.size
    resize_num = 4
    x = int(width/resize_num)
    zoom = 3
    size = (zoom*x, zoom*height)
    resized_roi = roi_area.resize(size)

    # resized_roi.save(f"dataset/roi_area/diaphragm({picture_num}).jpg")
    width, height = resized_roi.size
    # plt.imshow(im)
    # plt.show()
    # im = Image.open(f"./dataset/roi_area/roi_area_{picture_num}.jpg")
    step = 192
    original_peak_x = []
    original_peak_top_y = []
    original_peak_bottom_y = []

    original_valley_x = []
    original_valley_top_y = []
    original_valley_bottom_y = []

    for patch_num in range(0, ceil((width-height)/step)+1):
        print(f'now it\'s patch {patch_num}')

        #! 完整roi切片
        patch_left = step*patch_num
        patch_top = 0
        patch_right = height+step*patch_num
        patch_bottom = height
        input_patch = resized_roi.crop(
            (patch_left, patch_top, patch_right, patch_bottom))
        #! 语义分割
        img = input_patch.convert('L')
        img = img.resize((192, 192))
        numpy_input = ((np.array(img))[
                       :, :, np.newaxis] / 255.).astype(np.float32)
        input = torch.tensor((numpy_input.transpose((2, 0, 1)))[np.newaxis, :])

        preds = model(input)
        preds = preds.data.cpu().numpy()
        pred_mask = preds[0, 0, :, :]
        a = np.sum(pred_mask, axis=0)
        b = a.sum()
        # if patch_num == 1:
        #     print(f'!!!b is {b}')
        pred_mask = np.where(pred_mask > 0, 1, 0)
        # plt.imshow(pred_mask)
        # plt.savefig(f'dataset\patch\output\\roi_area_{i}_patch_{j+1}.jpg')
        # plt.close()
        #! 分割结果轮廓提取,保留最大面积的轮廓
        pred_mask_0 = (np.array(pred_mask, dtype='uint8'))
        pred_mask = cv2.medianBlur(pred_mask_0, 15)
        contours, _ = cv2.findContours(pred_mask, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            print(
                f'find no segged part in ./dataset/roi_area/roi_area_{picture_num}_patch_{patch_num+1}.jpg')
            break

        areas = []
        for contour in contours:
            areas.append(cv2.moments(contour)['m00'])
        areas = np.array(areas)
        index = np.argmax(areas)
        main_contour = contours[index]
        color = 2
        cv2.polylines(pred_mask, [main_contour], True, color, 1)
        # plt.subplot(1, 3, 1)
        # plt.imshow((np.array(img)))
        # plt.subplot(1, 3, 2)
        # plt.imshow(pred_mask_0)
        # plt.subplot(1, 3, 3)
        # plt.imshow(pred_mask)
        # plt.show()

        #! 原始上下沿提取
        length = 192
        x = np.arange(length)

        top_line = np.zeros(length)
        bottom_line = np.zeros(length)
        for i in range(0, length):
            for j in range(0, 192):
                if pred_mask[j, i] == color:
                    top_line[i] = j
                    break
            for j in range(191, 0, -1):
                if pred_mask[j, i] == color:
                    bottom_line[i] = j
                    break
        #! 获取上下沿有效区域起止点
        top_line_start = 0
        top_line_end = 0
        bottom_line_start = 0
        bottom_line_end = 0
        for i in range(length):
            if top_line[i] != 0:
                top_line_start = i
                break
        for i in range(191, 0, -1):
            if top_line[i] != 0:
                top_line_end = i
                break
        for i in range(length):
            if bottom_line[i] != 0:
                bottom_line_start = i
                break
        for i in range(191, 0, -1):
            if bottom_line[i] != 0:
                bottom_line_end = i
                break

        top_search_start = int((top_line_start+top_line_end)/2)
        bottom_search_start = int((bottom_line_start+bottom_line_end)/2)
        # 绘制上下沿的起点,终点,中点

        #! 分割区域边缘会凹凸不平
        #! 根据上下沿两端的差分,判断哪里是凹凸不平的地方,从而截断
        # TODO 差分的阈值是超参数
        top_cut_point_start = find_start_cut_point(
            top_line, top_line_start, top_search_start)
        top_cut_point_end = find_end_cut_point(
            top_line, top_line_end, top_search_start)
        bottom_cut_point_start = find_start_cut_point(
            bottom_line, bottom_line_start, bottom_search_start)
        bottom_cut_point_end = find_end_cut_point(
            bottom_line, bottom_line_end, bottom_search_start)
        # 画截断点位置和截断线
        cut_point_start = max(top_cut_point_start, bottom_cut_point_start)
        cut_point_end = min(top_cut_point_end, bottom_cut_point_end)

        #! 开始检测波峰波谷
        # 把bottom line的有效区域作为参考线,对其进行强烈平滑
        refer_line = np.zeros(length)
        for i in range(cut_point_start, cut_point_end):
            refer_line[i] = 1*(bottom_line[i]+top_line[i])
            # refer_line[i] = (bottom_line[i])

        # line_data[number, 0, :] = refer_line.copy()
        valid_line = refer_line.copy()
        # 0通道放peak(原图中向下突出的地方),1通道放valley(原图中向上凹陷的地方)
        label = np.zeros((2, length))
        N = 8
        p3 = np.zeros(length)
        box = refer_line[cut_point_start:cut_point_start+2*N].sum()
        for j in range(cut_point_start+N+1, cut_point_end-N-1):
            box = box-refer_line[j-N]+refer_line[j+N]
            # p3[j] = int(box) >> 4
            p3[j] = box/16

        # refer_line = p_bottom-p_top
        refer_line = p3

        add_line = np.zeros(length)
        for i in range(cut_point_start, cut_point_end):
            add_line[i] = (top_line[i]+bottom_line[i])
        # plt.plot(x, top_line, c='r')
        # plt.plot(x, bottom_line, c='g')
        # plt.plot(x, refer_line)
        # plt.plot(x, valid_line)

        exhale_thickness = []
        inhale_thickness = []
        exhale_peak = []
        inhale_peak = []
        # TODO 寻峰的平滑窗长与最值窗长也是超参数
        half_window = 10
        # ! 如果写if (picture_num == 2 & patch_num == 1)会进不去
        if ((picture_num == 2) & (patch_num == 4)):
            add_line = []
        for i in range(cut_point_start+half_window, cut_point_end-half_window):
            if ((refer_line[i] > refer_line[i-1]) & (refer_line[i] >= refer_line[i+1]) & (peak_judge(bottom_line, refer_line, i, cut_point_start, cut_point_end, half_window))):
                # 最值锚定,平台居中
                (index, max_start, max_end) = correct_peak(
                    i, bottom_line, half_window, cut_point_end-half_window)
                exhale_thickness.append(top_line[index]-bottom_line[index])
                exhale_peak.append(index)
                # plt.scatter(index, valid_line[index], marker='x', c='r')
                line_label[number, 0, max_start:max_end] = 1

            if ((refer_line[i] < refer_line[i-1]) & (refer_line[i] <= refer_line[i+1]) & (valley_judge(bottom_line, refer_line, i, cut_point_start, cut_point_end, half_window))):
                (index, min_start, min_end) = correct_valley(
                    i, bottom_line, half_window, cut_point_end-half_window)
                inhale_thickness.append(top_line[index]-bottom_line[index])
                inhale_peak.append(index)
                # plt.scatter(index, valid_line[index], marker='x', c='b')
                line_label[number, 1, min_start:min_end] = 1

        # plt.plot((cut_point_start+half_window, cut_point_start+half_window),
        #          (bottom_line[cut_point_start+half_window], top_line[cut_point_start+half_window]), c='b', marker='*', linestyle='--')
        # plt.plot((cut_point_end-half_window, cut_point_end-half_window),
        #          (bottom_line[cut_point_end-half_window], top_line[cut_point_end-half_window]), c='b', marker='*', linestyle='--')
        # plt.show()

        #! 切片坐标和原图坐标的映射变换
        if (len(exhale_peak) != 0):
            for i in range(len(exhale_peak)):
                original_peak_x.append(
                    w_start[picture_num]+((step*patch_num+exhale_peak[i]*(576/192))*resize_num/zoom))
                original_peak_top_y.append(
                    (top_line[exhale_peak[i]]*576./192.)/zoom+roi_depth-half_roi_height)
                original_peak_bottom_y.append(
                    (bottom_line[exhale_peak[i]]*576./192.)/zoom+roi_depth-half_roi_height)

        if (len(inhale_peak) != 0):
            for i in range(len(inhale_peak)):
                original_valley_x.append(
                    w_start[picture_num]+((step*patch_num+inhale_peak[i]*(576/192))*resize_num/zoom))
                original_valley_top_y.append(
                    (top_line[inhale_peak[i]]*576./192.)/zoom+roi_depth-half_roi_height)
                original_valley_bottom_y.append(
                    (bottom_line[inhale_peak[i]]*576./192.)/zoom+roi_depth-half_roi_height)
        number += 1

    #! 波峰聚类
    (refined_peak_x, refined_peak_top_y, refined_peak_bottom_y) = peaks_clustering(
        original_peak_x.copy(), original_peak_top_y.copy(), original_peak_bottom_y.copy())
    #! 波谷聚类
    (refined_valley_x, refined_valley_top_y, refined_valley_bottom_y) = peaks_clustering(
        original_valley_x.copy(), original_valley_top_y.copy(), original_valley_bottom_y.copy())

    raw_image = Image.open(
        f".\\dataset\\raw_image\\diaphragm ({picture_num}).jpg")
    # dataset\raw_image\useful (1).jpg
    w, h = raw_image.size
    plt.imshow(raw_image)

    inhale_thickness = []
    for i in range(len(refined_peak_x)):
        plt.scatter(refined_peak_x[i],
                    refined_peak_top_y[i]-1, marker='+', c='r')
        plt.scatter(refined_peak_x[i],
                    refined_peak_bottom_y[i]-1, marker='+', c='r')
        x_place = refined_peak_x[i]-60
        y_place = refined_peak_bottom_y[i]+20
        inhale_thickness.append(
            float(refined_peak_bottom_y[i]-refined_peak_top_y[i]))
        plt.text(x_place, y_place,
                 f'inhale: {inhale_thickness[i]:.2f}', c='r', fontsize=12)

    exhale_thickness = []
    for i in range(len(refined_valley_x)):
        plt.scatter(refined_valley_x[i],
                    refined_valley_top_y[i]-1, marker='+', c='y')
        plt.scatter(refined_valley_x[i],
                    refined_valley_bottom_y[i]-1, marker='+', c='y')
        x_place = refined_valley_x[i]-60
        y_place = refined_valley_bottom_y[i]+20
        exhale_thickness.append(
            float(refined_valley_bottom_y[i]-refined_valley_top_y[i]))
        plt.text(x_place, y_place,
                 f'exhale: {exhale_thickness[i]:.2f}', c='y', fontsize=12)
    average_inhale_thickness = sum(inhale_thickness)/len(inhale_thickness)
    average_exhale_thickness = sum(exhale_thickness)/len(exhale_thickness)
    plt.text(int(w/2)-160, h-180,
             f'average inhale: {average_inhale_thickness:.2f}', c='g', fontsize=14)
    plt.text(int(w/2)-160, h-140,
             f'average exhale: {average_exhale_thickness:.2f}', c='g', fontsize=14)
    plt.show()

np.save('./dataset/lines/line_data.npy', line_data)
np.save('./dataset/lines/line_label.npy', line_label)

print(f'number = {number}')

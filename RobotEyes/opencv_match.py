# coding: utf-8
import base64
import io
import os
import re
import typing
from typing import Union
import cv2
import imutils
import numpy as np
import requests
from PIL import Image, ImageDraw
import matplotlib.pylab as plt
from skimage.metrics import structural_similarity

from . import Log

DEBUG = False
ENABLE_CALC_TIME = False

ImageType = typing.Union[np.ndarray, Image.Image]

compare_ssim = structural_similarity


def color_bgr2gray(image: ImageType):
    """ change color image to gray
    Returns:
        opencv-image
    """
    if ispil(image):
        image = pil2cv(image)

    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def template_ssim(image_a: ImageType, image_b: ImageType):
    """
    Refs:
        https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
    """
    a = color_bgr2gray(image_a)
    b = color_bgr2gray(image_b)  # template (small)
    res = cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return max_val


def cv2crop(im: np.ndarray, bounds: tuple = None):
    if not bounds:
        return im
    assert len(bounds) == 4

    lx, ly, rx, ry = bounds
    crop_img = im[ly:ry, lx:rx]
    return crop_img


def compare_ssim(image_a: ImageType, image_b: ImageType, full=False, bounds=None):
    a = color_bgr2gray(image_a)
    b = color_bgr2gray(image_b)  # template (small)
    ca = cv2crop(a, bounds)
    cb = cv2crop(b, bounds)
    return structural_similarity(ca, cb, full=full)


def compare_ssim_debug(image_a: ImageType, image_b: ImageType, color=(255, 0, 0)):
    """
    Args:
        image_a, image_b: opencv image or PIL.Image
        color: (r, g, b) eg: (255, 0, 0) for red
    Refs:
        https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
    """
    ima, imb = conv2cv(image_a), conv2cv(image_b)
    score, diff = compare_ssim(ima, imb, full=True)
    diff = (diff * 255).astype('uint8')
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cv2color = tuple(reversed(color))
    im = ima.copy()
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(im, (x, y), (x + w, y + h), cv2color, 2)
    # todo: show image
    cv2pil(im).show()
    return im


def show_image(im: Union[np.ndarray, Image.Image]):
    pilim = conv2pil(im)
    pilim.show()


def pil2cv(pil_image) -> np.ndarray:
    """ Convert from pillow image to opencv """
    # convert PIL to OpenCV
    pil_image = pil_image.convert('RGB')
    cv2_image = np.array(pil_image)
    # Convert RGB to BGR
    cv2_image = cv2_image[:, :, ::-1].copy()
    return cv2_image


def pil2base64(pil_image, format="JPEG") -> str:
    """ Convert pillow image to base64 """
    buf = io.BytesIO()
    pil_image.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def cv2pil(cv_image):
    """ Convert opencv to pillow image """
    return Image.fromarray(cv_image[:, :, ::-1].copy())


def iscv2(im):
    return isinstance(im, np.ndarray)


def ispil(im):
    return isinstance(im, Image.Image)


def conv2cv(im: Union[np.ndarray, Image.Image]) -> np.ndarray:
    if iscv2(im):
        return im
    if ispil(im):
        return pil2cv(im)
    raise TypeError("Unknown image type:", type(im))


def conv2pil(im: Union[np.ndarray, Image.Image]) -> Image.Image:
    if ispil(im):
        return im
    elif iscv2(im):
        return cv2pil(im)
    else:
        raise TypeError(f"Unknown image type: {type(im)}")


def _open_data_url(data, flag=cv2.IMREAD_COLOR):
    pos = data.find('base64,')
    if pos == -1:
        raise IOError("data url is invalid, head %s" % data[:20])

    pos += len('base64,')
    raw_data = base64.decodestring(data[pos:])
    image = np.asarray(bytearray(raw_data), dtype="uint8")
    image = cv2.imdecode(image, flag)
    return image


def _open_image_url(url: str, flag=cv2.IMREAD_COLOR):
    """ download the image, convert it to a NumPy array, and then read
    it into OpenCV format """
    content = requests.get(url).content
    image = np.asarray(bytearray(content), dtype="uint8")
    image = cv2.imdecode(image, flag)
    return image


def draw_point(im: Image.Image, x: int, y: int) -> Image.Image:
    """
    Mark position to show which point clicked
    Args:
        im: pillow.Image
    """
    draw = ImageDraw.Draw(im)
    w, h = im.size
    draw.line((x, 0, x, h), fill='red', width=5)
    draw.line((0, y, w, y), fill='red', width=5)
    r = min(im.size) // 40
    draw.ellipse((x - r, y - r, x + r, y + r), fill='red')
    r = min(im.size) // 50
    draw.ellipse((x - r, y - r, x + r, y + r), fill='white')
    del draw
    return im


def imread(data) -> np.ndarray:
    """
    Args:
        data: local path or http url or data:image/base64,xxx

    Returns:
        opencv image

    Raises:
        IOError
    """
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, Image.Image):
        return pil2cv(data)
    elif data.startswith('data:image/'):
        return _open_data_url(data)
    elif re.match(r'^https?://', data):
        return _open_image_url(data)
    elif os.path.isfile(data):
        im = cv2.imread(data)
        if im is None:
            raise IOError("Image format error: %s" % data)
        return im

    raise IOError("image read invalid data: %s" % data)


class UIMatcher(object):
    @staticmethod
    def RotateClockWise90(img):
        trans_img = cv2.transpose(img)
        new_img = cv2.flip(trans_img, 0)
        return new_img

    @staticmethod
    def template_match(screen, template_path):
        """
        模板匹配
        :param screen: 屏幕截图
        :param template_path: 模板路径
        :return: {'r': 相似度, 'x': x坐标, 'y': y坐标}
        """
        res = {'r': 0.0,
               'x': 0,
               'y': 0}

        # 旋转屏幕截图
        if screen.shape[0] > screen.shape[1]:
            screen = UIMatcher.RotateClockWise90(screen)
        height, width = screen.shape[0:2:1]
        raw_screen = screen.copy()  # DEBUG
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        # 读取模板
        template = cv2.imdecode(np.fromfile(template_path, dtype=np.uint8),
                                cv2.IMREAD_GRAYSCALE)
        # 模板匹配
        result = cv2.matchTemplate(template, screen, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        # 计算坐标
        h, w = template.shape[0:2:1]
        res['x'] = max_loc[0] + w / 2
        res['y'] = max_loc[1] + h / 2
        res['r'] = max_val
        # DEBUG
        if DEBUG:
            UIMatcher._plot_boundary(raw_screen, template, max_loc)

        return res

    @staticmethod
    def _plot_boundary(screen, template, max_loc):
        """
        绘制模板在屏幕上的区域
        :param screen: 屏幕截图
        :param template: 模板图像
        :param max_loc: 匹配信息
        :return: void
        """
        h, w = template.shape[0:2:1]
        cv2.rectangle(screen,
                      (int(max_loc[0]), int(max_loc[1])),
                      (int(max_loc[0] + w), int(max_loc[1] + h)),
                      (0, 0, 255), 2)
        plt.cla()
        matched_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        plt.imshow(matched_screen)
        plt.pause(0.01)

    @staticmethod
    def _multi_scale_template_match(screen, template_path):
        """
        :param screen:
        :param template_path:
        :return:
        """
        # result = UIMatcher.template_match(screen, template_path)
        found = None
        # 循环遍历不同的尺度
        for scale in np.linspace(1.4, 0.6, 50)[::-1]:
            # 根据尺度大小对输入图片进行裁剪
            resized = imutils.resize(screen, width=int(screen.shape[1] * scale))
            result = UIMatcher.template_match(resized, template_path)
            # 如果发现一个新的关联值则进行更新
            if found is None or result['r'] > found['r']:
                found = result

        return found

    @staticmethod
    def multi_scale_template_match(screen_path, template_path, min_scale=0.8, max_scale=1.4, step=40, save_dir=None):
        """
        多尺度模板匹配
        :param screen: 屏幕截图
        :param template_path: 模板路径
        :param min_scale: 比例下限
        :param max_scale: 比例上限
        :param step: 比例划分次数（数字越大准确度越高，但是耗时）
        :return: {'r': 相似度, 'x': x坐标, 'y': y坐标}
        """
        if ENABLE_CALC_TIME:
            import time
            start_time = time.time()
            print(start_time)
        # 旋转屏幕截图
        screen = imread(screen_path)
        if screen.shape[0] > screen.shape[1]:
            screen = UIMatcher.RotateClockWise90(screen)
        raw_screen = screen.copy()  # DEBUG
        # 读取模板图片
        template = imread(template_path)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        # 执行边缘检测
        template = cv2.Canny(template, 50, 200)
        (tH, tW) = template.shape[:2]
        # 显示模板
        if DEBUG:
            plt.imshow(template)
            plt.pause(0.01)
        # 读取测试图片并将其转化为灰度图片
        gray = cv2.cvtColor(raw_screen, cv2.COLOR_BGR2GRAY)
        found = None
        # 循环遍历不同的尺度
        for scale in np.linspace(max_scale, min_scale, step)[::-1]:
            # 根据尺度大小对输入图片进行裁剪
            resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])
            # 如果裁剪之后的图片小于模板的大小直接退出
            if resized.shape[0] < tH or resized.shape[1] < tW:
                continue
            # 首先进行边缘检测，然后执行模板检测，接着获取最小外接矩形
            edged = cv2.Canny(resized, 50, 200)
            # plt.imshow(edged)
            result = cv2.matchTemplate(template, edged, cv2.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            # 结果可视化
            if DEBUG:  # 绘制矩形框并显示结果
                # clone = np.dstack([edged, edged, edged])
                # cv2.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
                # plt.imshow(clone)
                # plt.pause(0.01)
                Log.color_log.info("> 比例：%.2f -> 相似度：%.2f", scale, maxVal)

            # 如果发现一个新的关联值则进行更新
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)

        # 计算测试图片中模板所在的具体位置，即左上角和右下角的坐标值，并乘上对应的裁剪因子
        (_, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

        # 绘制结果
        if save_dir:
            matched_screen = cv2.rectangle(raw_screen, (startX, startY), (endX, endY), (0, 0, 255), 2)
            matched_screen = cv2.cvtColor(matched_screen, cv2.COLOR_BGR2RGB)
            if DEBUG:
                plt.imshow(matched_screen)
                plt.pause(0.01)
            try:
                plt.imsave(save_dir + "/res_" + template_path, matched_screen)
            except FileNotFoundError:
                Log.color_log.debug(f'保存文件出错：{save_dir + "/res_" + template_path}')


        if ENABLE_CALC_TIME:
            end_time = time.time()
            print(end_time)
            Log.color_log.debug("耗时：%.2f", end_time - start_time)

        # 计算坐标
        res = dict()
        res['x'] = (startX + endX) / 2
        res['y'] = (startY + endY) / 2
        res['r'] = found[0]
        res['x0'], res['x1'], res['y0'], res['y1'] = startX, endX, startY, endY
        Log.color_log.debug(res)

        return res

if __name__ == '__main__':
    res = UIMatcher.multi_scale_template_match('/Users/orionc/workspace/dev_project/RobotEyes/RobotEyes/testfull.png', "http://oldboy.run:9000/myserver/output/ff0b5b17-a524-4ee3-911d-fb0954080603.png", min_scale=0.5, max_scale=2)
    print(res)
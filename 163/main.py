def draw_img(image):
  plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  plt.show()

import numpy as np # Импорт модуля numpy
import cv2 # импорт OpenCV
from matplotlib import pyplot as plt

# Загрузка изображения
image = cv2.imread('Morph.jpg', cv2.IMREAD_GRAYSCALE)



# Создание структурных элементов
kernel_cross_3x3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
kernel_cross_5x5 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
kernel_rect_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel_rect_5x5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel_ellipse_3x3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_ellipse_5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def make_all_operation(path, folder):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Применение морфологических операций с различными структурными элементами и итерациями
    for iterations in [1, 3, 5]:
        dilations = [
            ('Cross 3x3', cv2.dilate(image, kernel_cross_3x3, iterations=iterations)),
            ('Cross 5x5', cv2.dilate(image, kernel_cross_5x5, iterations=iterations)),
            ('Rectangle 3x3', cv2.dilate(image, kernel_rect_3x3, iterations=iterations)),
            ('Rectangle 5x5', cv2.dilate(image, kernel_rect_5x5, iterations=iterations)),
            ('Ellipse 3x3', cv2.dilate(image, kernel_ellipse_3x3, iterations=iterations)),
            ('Ellipse 5x5', cv2.dilate(image, kernel_ellipse_5x5, iterations=iterations))
        ]

        erosions = [
            ('Cross 3x3', cv2.erode(image, kernel_cross_3x3, iterations=iterations)),
            ('Cross 5x5', cv2.erode(image, kernel_cross_5x5, iterations=iterations)),
            ('Rectangle 3x3', cv2.erode(image, kernel_rect_3x3, iterations=iterations)),
            ('Rectangle 5x5', cv2.erode(image, kernel_rect_5x5, iterations=iterations)),
            ('Ellipse 3x3', cv2.erode(image, kernel_ellipse_3x3, iterations=iterations)),
            ('Ellipse 5x5', cv2.erode(image, kernel_ellipse_5x5, iterations=iterations))
        ]

        # Сохранение результатов
        for name, result in dilations:
            cv2.imwrite(f'{folder}/dilation_{name}_iterations_{iterations}.jpg', result)

        for name, result in erosions:
            cv2.imwrite(f'{folder}/erosion_{name}_iterations_{iterations}.jpg', result)

make_all_operation('Morph.jpg', 'morph')

# сравним операции закрытия с результатамипоследовательного применения дилатации и эрозии

closing_op_result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_rect_5x5)
er_dil_result = cv2.erode(cv2.dilate(image, kernel_rect_5x5), kernel_rect_5x5)

cv2.imwrite('2/close.jpg', closing_op_result) # они одинаковые
cv2.imwrite('2/erdil.jpg', er_dil_result)

# сравним операции открытия с результатамипоследовательного применения  эрозии и дилатации

closing_op_result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_rect_5x5)
er_dil_result = cv2.dilate(cv2.erode(image, kernel_rect_5x5), kernel_rect_5x5)

cv2.imwrite('2/open.jpg', closing_op_result) # они одинаковые
cv2.imwrite('2/diler.jpg', er_dil_result)

# ищем границы
edges = image - cv2.erode(image, kernel_rect_5x5)
cv2.imwrite('2/edges.jpg', edges)

# ищем лучший элемент для morph2.bmp

make_all_operation('morph2.bmp', 'morph2')

# лучше все справился эллипс для morph2. Для morph в принципе и эллипс и крест одинаково себя показали


# Применение бинаризации, часть 6
def get_image_between_thresholds(image, lower, upper):

    # Создание маски для пикселей в указанном диапазоне
    mask = cv2.inRange(image, lower, upper)

    # Применение маски к изображению
    return cv2.bitwise_and(image, image, mask=mask)

image = cv2.imread('morph3.bmp', cv2.IMREAD_GRAYSCALE)

binary_image1 = get_image_between_thresholds(image, 90, 100)
edges = binary_image1 - cv2.erode(binary_image1, kernel_rect_5x5)
cv2.imwrite('2/edges_binary_1.jpg', edges)

binary_image2 = get_image_between_thresholds(image, 10, 90)
edges = binary_image2 - cv2.erode(binary_image2, kernel_rect_5x5)
cv2.imwrite('2/edges_binary_2.jpg', edges)

binary_image3 = get_image_between_thresholds(image, 101, 150)
edges = binary_image3 - cv2.erode(binary_image3, kernel_rect_5x5)
cv2.imwrite('2/edges_binary_3.jpg', edges)

binary_image4 = get_image_between_thresholds(image, 151, 255)
edges = binary_image4 - cv2.erode(binary_image4, kernel_rect_5x5)
cv2.imwrite('2/edges_binary_4.jpg', edges)


# часть 7
# Загрузка изображения
img = cv2.imread('Fingerprint.jpg', 0)  # Загружаем как черно-белое (grayscale) изображение

# Бинаризация изображения
_, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Эрозия и дилатация
kernel = np.ones((3, 3), np.uint8)
eroded_img = cv2.erode(binary_img, kernel, iterations=1)
dilated_img = cv2.dilate(eroded_img, kernel, iterations=1)

# Открытие и закрытие
opened_img = cv2.morphologyEx(dilated_img, cv2.MORPH_OPEN, kernel)
closed_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, kernel)

# Подавление шума
denoised_img = cv2.medianBlur(closed_img, 5)

# сохранение результата
cv2.imwrite('2/fingerprint.jpg', denoised_img)


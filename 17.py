import cv2
import numpy as np


# Функция для загрузки изображения и выполнения операций
def process_image(filename):
    # Загрузка изображения
    image = cv2.imread(filename)

    # Полутоновое изображение
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Изображение высотой 400 пикселей
    resized_image = cv2.resize(image, (int(400 * image.shape[1] / image.shape[0]), 400))

    # Поворот на 180 градусов
    rotated180_image = cv2.rotate(image, cv2.ROTATE_180)

    # Поворот на 90 градусов
    rotated90_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # Вертикальное отражение
    flipped_vertical_image = cv2.flip(image, 0)

    # Горизонтальное отражение
    flipped_horizontal_image = cv2.flip(image, 1)

    # Определение объекта и подпись к нему
    # Пример: лошадь на изображении "лошадь.jpg"
    # (Здесь должен быть код для обнаружения объекта на изображении и добавления подписи)

    # Вывод результатов
    cv2.imshow("Original Image", image)
    cv2.imshow("Grayscale Image", gray_image)
    cv2.imshow("Resized Image", resized_image)
    cv2.imshow("Rotated 180 Image", rotated180_image)
    cv2.imshow("Rotated 90 Image", rotated90_image)
    cv2.imshow("Flipped Vertical Image", flipped_vertical_image)
    cv2.imshow("Flipped Horizontal Image", flipped_horizontal_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Вызов функции для каждого изображения
# process_image("лошадь.jpg")
# process_image("bus.jpg")
# process_image("car.jpg")
# process_image("fox.jpg")
# process_image("tennis_ball.jpg")
# process_image("Айвазовский.jpg")

def find_object(image_path, color_low, color_high):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #Преобразуем в HSV
    only_obj_hsv = cv2.inRange(hsv, color_low, color_high)
    cv2.imshow("only cat", only_obj_hsv)

    moments = cv2.moments(only_obj_hsv, 1) # получим моменты
    x_moment = moments['m10']
    y_moment = moments['m01']
    area = moments['m00']
    x = int(x_moment / area)
    y = int(y_moment / area) # и выведем текст на изображение
    cv2.putText(image, image_path, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    cv2.imshow('показываем объект', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# find_object('fox.jpg', (7,40,60), (18,255,200))

find_object('car.jpg', (150, 44, 39), (255,50,50))